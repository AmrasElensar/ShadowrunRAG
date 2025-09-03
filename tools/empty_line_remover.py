#!/usr/bin/env python3
"""
Remove lines that contain ONLY line endings (completely empty lines).
Preserves intentional paragraph breaks but removes artifact empty lines.
"""

from pathlib import Path
import re


def analyze_empty_lines(content: str) -> dict:
    """Analyze empty line patterns in the content"""
    lines = content.split('\n')

    stats = {
        'total_lines': len(lines),
        'completely_empty': 0,
        'whitespace_only': 0,
        'has_content': 0,
        'consecutive_empty_blocks': []
    }

    consecutive_empty = 0
    empty_block_start = -1

    for i, line in enumerate(lines):
        if len(line) == 0:
            # Completely empty line
            stats['completely_empty'] += 1
            if consecutive_empty == 0:
                empty_block_start = i + 1  # 1-based line number
            consecutive_empty += 1
        elif len(line.strip()) == 0:
            # Whitespace-only line
            stats['whitespace_only'] += 1
            if consecutive_empty == 0:
                empty_block_start = i + 1
            consecutive_empty += 1
        else:
            # Line has actual content
            stats['has_content'] += 1
            if consecutive_empty > 0:
                stats['consecutive_empty_blocks'].append({
                    'start_line': empty_block_start,
                    'end_line': i,  # 1-based
                    'count': consecutive_empty
                })
            consecutive_empty = 0

    # Handle case where file ends with empty lines
    if consecutive_empty > 0:
        stats['consecutive_empty_blocks'].append({
            'start_line': empty_block_start,
            'end_line': len(lines),
            'count': consecutive_empty
        })

    return stats


def smart_empty_line_removal(content: str) -> str:
    """
    Remove excessive empty lines while preserving document structure.

    Rules:
    - Remove all lines that are completely empty (length 0)
    - Convert lines with only whitespace to completely empty
    - Preserve at most 2 consecutive empty lines (for section breaks)
    - Keep 1 empty line before headers
    """
    lines = content.split('\n')
    cleaned_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if line is empty or whitespace-only
        if len(line.strip()) == 0:
            # Count consecutive empty/whitespace lines
            empty_count = 0
            j = i
            while j < len(lines) and len(lines[j].strip()) == 0:
                empty_count += 1
                j += 1

            # Check what comes after the empty block
            next_line = lines[j] if j < len(lines) else ""
            is_next_header = (next_line.startswith('#') or
                              re.match(r'^[A-Z\s\(\)\-&,\']{10,}$', next_line.strip()))

            # Decide how many empty lines to keep
            if empty_count == 1:
                # Single empty line - keep it (paragraph break)
                cleaned_lines.append('')
            elif empty_count == 2:
                # Two empty lines - keep both (section break)
                cleaned_lines.extend(['', ''])
            elif is_next_header:
                # Before a header - keep 2 empty lines max
                cleaned_lines.extend(['', ''])
            else:
                # Multiple empty lines in content - reduce to 1
                cleaned_lines.append('')

            # Skip all the empty lines we just processed
            i = j
        else:
            # Line has content - keep it as-is
            cleaned_lines.append(line)
            i += 1

    return '\n'.join(cleaned_lines)


def aggressive_empty_line_removal(content: str) -> str:
    """
    More aggressive - removes ALL completely empty lines.
    Use this if smart removal isn't enough.
    """
    lines = content.split('\n')
    return '\n'.join(line for line in lines if len(line.strip()) > 0)


def process_file_empty_lines(input_file: str, output_file: str = None, aggressive: bool = False) -> None:
    """Process a file to remove empty lines"""

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"Analyzing empty lines in: {input_file}")
        stats = analyze_empty_lines(content)

        print(f"  Total lines: {stats['total_lines']}")
        print(f"  Completely empty: {stats['completely_empty']}")
        print(f"  Whitespace only: {stats['whitespace_only']}")
        print(f"  Has content: {stats['has_content']}")

        if stats['consecutive_empty_blocks']:
            print(f"  Consecutive empty blocks: {len(stats['consecutive_empty_blocks'])}")
            for block in stats['consecutive_empty_blocks'][:5]:  # Show first 5
                print(f"    Lines {block['start_line']}-{block['end_line']}: {block['count']} empty lines")

        if stats['completely_empty'] == 0 and stats['whitespace_only'] == 0:
            print("  ✓ No empty lines to remove")
            return

        print(f"Processing with {'aggressive' if aggressive else 'smart'} mode...")

        if aggressive:
            cleaned_content = aggressive_empty_line_removal(content)
        else:
            cleaned_content = smart_empty_line_removal(content)

        new_stats = analyze_empty_lines(cleaned_content)

        output_path = output_file if output_file else input_file
        with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(cleaned_content)

        print(f"  ✓ Removed {stats['total_lines'] - new_stats['total_lines']} empty lines")
        print(f"  Final: {new_stats['total_lines']} lines ({new_stats['completely_empty']} empty)")
        print(f"  Saved to: {output_path}")

    except Exception as e:
        print(f"✗ Error processing {input_file}: {e}")


def main():
    """Main execution function"""
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python empty_line_remover.py <file>")
        print("  python empty_line_remover.py <file> --aggressive")
        print("  python empty_line_remover.py <directory>")
        print("  python empty_line_remover.py <directory> --aggressive")
        print()
        print("Modes:")
        print("  default: Smart removal (preserves document structure)")
        print("  --aggressive: Removes ALL empty lines")
        sys.exit(1)

    input_path = sys.argv[1]
    aggressive = '--aggressive' in sys.argv

    path_obj = Path(input_path)

    if path_obj.is_file():
        process_file_empty_lines(str(path_obj), aggressive=aggressive)
    elif path_obj.is_dir():
        md_files = list(path_obj.glob("*.md"))
        print(f"Found {len(md_files)} markdown files")
        for file_path in md_files:
            process_file_empty_lines(str(file_path), aggressive=aggressive)
            print()
    else:
        print(f"Path not found: {input_path}")


if __name__ == "__main__":
    main()