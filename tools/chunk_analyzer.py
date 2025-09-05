#!/usr/bin/env python3
"""
Analyze current chunking quality to identify improvement opportunities.

Save this as: tools/chunk_analyzer.py
Run from project root: python tools/chunk_analyzer.py
"""

import chromadb
from chromadb.config import Settings
import json
import statistics
import re
from pathlib import Path
from collections import Counter, defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkAnalyzer:
    def __init__(self, chroma_path: str = "data/chroma_db", collection_name: str = "shadowrun_docs"):
        self.chroma_path = Path(chroma_path)

        if not self.chroma_path.exists():
            logger.error(f"ChromaDB not found at {chroma_path}")
            return

        self.client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=Settings(anonymized_telemetry=False)
        )

        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"✅ Connected to collection '{collection_name}'")
        except Exception as e:
            logger.error(f"❌ Could not connect to collection: {e}")
            return

    def analyze_chunks(self):
        """Comprehensive chunk quality analysis."""

        # Get all chunks
        try:
            all_chunks = self.collection.get(include=['documents', 'metadatas'])
            total_chunks = len(all_chunks['documents'])
            logger.info(f"Analyzing {total_chunks} chunks...")
        except Exception as e:
            logger.error(f"Failed to retrieve chunks: {e}")
            return

        if total_chunks == 0:
            logger.warning("No chunks found in collection")
            return

        # Initialize analysis results
        analysis = {
            "total_chunks": total_chunks,
            "size_analysis": {},
            "classification_analysis": {},
            "content_quality": {},
            "semantic_boundaries": {},
            "specific_issues": {}
        }

        documents = all_chunks['documents']
        metadatas = all_chunks['metadatas'] or [{}] * len(documents)

        # 1. CHUNK SIZE ANALYSIS
        logger.info("📊 Analyzing chunk sizes...")

        word_counts = [len(doc.split()) for doc in documents]
        char_counts = [len(doc) for doc in documents]

        analysis["size_analysis"] = {
            "word_count_stats": {
                "min": min(word_counts),
                "max": max(word_counts),
                "mean": statistics.mean(word_counts),
                "median": statistics.median(word_counts),
                "std_dev": statistics.stdev(word_counts) if len(word_counts) > 1 else 0
            },
            "char_count_stats": {
                "min": min(char_counts),
                "max": max(char_counts),
                "mean": statistics.mean(char_counts),
                "median": statistics.median(char_counts)
            },
            "size_distribution": self._get_size_distribution(word_counts)
        }

        # 2. CLASSIFICATION ANALYSIS
        logger.info("🏷️  Analyzing classifications...")

        primary_sections = [meta.get('primary_section', 'Unknown') for meta in metadatas]
        document_types = [meta.get('document_type', 'Unknown') for meta in metadatas]
        editions = [meta.get('edition', 'Unknown') for meta in metadatas]

        analysis["classification_analysis"] = {
            "primary_sections": dict(Counter(primary_sections)),
            "document_types": dict(Counter(document_types)),
            "editions": dict(Counter(editions)),
            "section_dominance": self._calculate_section_dominance(primary_sections)
        }

        # 3. CONTENT QUALITY ANALYSIS
        logger.info("🔍 Analyzing content quality...")

        analysis["content_quality"] = {
            "empty_chunks": sum(1 for doc in documents if len(doc.strip()) < 50),
            "header_only_chunks": sum(1 for doc in documents if self._is_header_only(doc)),
            "incomplete_sentences": sum(1 for doc in documents if self._has_incomplete_sentences(doc)),
            "example_vs_rules": self._analyze_examples_vs_rules(documents),
            "cross_reference_issues": self._find_cross_reference_issues(documents)
        }

        # 4. SEMANTIC BOUNDARY ANALYSIS
        logger.info("🧩 Analyzing semantic boundaries...")

        analysis["semantic_boundaries"] = {
            "split_mid_sentence": sum(1 for doc in documents if self._splits_mid_sentence(doc)),
            "split_mid_table": sum(1 for doc in documents if self._splits_mid_table(doc)),
            "orphaned_headers": sum(1 for doc in documents if self._is_orphaned_header(doc)),
            "missing_context": sum(1 for doc in documents if self._missing_context(doc))
        }

        # 5. SPECIFIC ISSUES FOR YOUR QUERY
        logger.info("⚡ Analyzing specific issues (taser/stun damage)...")

        analysis["specific_issues"] = self._analyze_taser_stun_chunks(documents, metadatas)

        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)

        return analysis

    def _get_size_distribution(self, word_counts):
        """Analyze distribution of chunk sizes."""
        bins = {
            "very_small (0-100)": sum(1 for x in word_counts if x < 100),
            "small (100-300)": sum(1 for x in word_counts if 100 <= x < 300),
            "medium (300-600)": sum(1 for x in word_counts if 300 <= x < 600),
            "large (600-1000)": sum(1 for x in word_counts if 600 <= x < 1000),
            "very_large (1000+)": sum(1 for x in word_counts if x >= 1000)
        }
        return bins

    def _calculate_section_dominance(self, primary_sections):
        """Calculate how dominant the top sections are."""
        section_counts = Counter(primary_sections)
        total = len(primary_sections)

        if not section_counts:
            return {"error": "No sections found"}

        top_3 = section_counts.most_common(3)
        top_3_percentage = sum(count for _, count in top_3) / total * 100

        return {
            "top_3_sections": top_3,
            "top_3_percentage": round(top_3_percentage, 1),
            "unique_sections": len(section_counts),
            "classification_diversity": round(len(section_counts) / total * 100, 1)
        }

    def _is_header_only(self, text):
        """Check if chunk is mostly just headers."""
        lines = text.strip().split('\n')
        non_header_lines = [line for line in lines if not line.strip().startswith('#')]
        content_lines = [line for line in non_header_lines if len(line.strip()) > 10]
        return len(content_lines) <= 2

    def _has_incomplete_sentences(self, text):
        """Check for chunks that end mid-sentence."""
        text = text.strip()
        if not text:
            return True

        # Simple heuristic: text should end with proper punctuation
        last_char = text[-1]
        return last_char not in '.!?'

    def _analyze_examples_vs_rules(self, documents):
        """Identify chunks that are examples vs actual rules."""
        examples = 0
        rules = 0
        unclear = 0

        for doc in documents:
            doc_lower = doc.lower()

            # Example indicators
            example_indicators = ['example:', 'for example', 'e.g.', 'suppose', 'let\'s say', 'imagine']
            # Rule indicators
            rule_indicators = ['test:', 'dice pool:', 'threshold:', 'modifier:', 'when a character', 'roll', 'add',
                               'subtract']

            has_example = any(indicator in doc_lower for indicator in example_indicators)
            has_rule = any(indicator in doc_lower for indicator in rule_indicators)

            if has_example and not has_rule:
                examples += 1
            elif has_rule and not has_example:
                rules += 1
            else:
                unclear += 1

        return {
            "examples_only": examples,
            "rules_only": rules,
            "unclear_or_mixed": unclear,
            "example_percentage": round(examples / len(documents) * 100, 1)
        }

    def _find_cross_reference_issues(self, documents):
        """Find chunks with broken cross-references."""
        issues = 0
        for doc in documents:
            # Look for incomplete references
            if re.search(r'see page \d+|see chapter \d+|refer to|as described in', doc.lower()):
                if 'MISSING' in doc or len(doc) < 200:  # Short chunks with references are suspicious
                    issues += 1
        return issues

    def _splits_mid_sentence(self, text):
        """Check if chunk starts or ends mid-sentence."""
        lines = text.strip().split('\n')
        if not lines:
            return False

        first_line = lines[0].strip()
        last_line = lines[-1].strip()

        # Check if first line starts mid-sentence (lowercase start, no proper noun indicators)
        starts_mid = (first_line and
                      first_line[0].islower() and
                      not first_line.startswith('#') and
                      len(first_line) > 10)

        # Check if last line ends mid-sentence
        ends_mid = (last_line and
                    not last_line.endswith(('.', '!', '?', ':', ';')) and
                    len(last_line) > 5)

        return starts_mid or ends_mid

    def _splits_mid_table(self, text):
        """Check if chunk splits a table."""
        lines = text.strip().split('\n')
        table_lines = [line for line in lines if '|' in line and line.count('|') > 1]

        # If chunk has table lines but doesn't start/end with table headers, it might be split
        if table_lines:
            has_header = any(
                '---' in line or line.strip().startswith('|') and line.strip().endswith('|') for line in lines[:3])
            return not has_header and len(table_lines) > 0
        return False

    def _is_orphaned_header(self, text):
        """Check if chunk is just a header with minimal content."""
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        if not lines:
            return False

        header_lines = [line for line in lines if line.startswith('#')]
        content_lines = [line for line in lines if not line.startswith('#') and len(line) > 5]

        return len(header_lines) >= 1 and len(content_lines) <= 1

    def _missing_context(self, text):
        """Check if chunk lacks sufficient context to be understood."""
        # Very simple heuristic: chunks under 100 chars that don't contain key context words
        if len(text) > 100:
            return False

        context_words = ['shadowrun', 'test', 'roll', 'dice', 'attribute', 'skill', 'character', 'damage', 'armor']
        return not any(word in text.lower() for word in context_words)

    def _analyze_taser_stun_chunks(self, documents, metadatas):
        """Specific analysis for taser/stun damage query."""

        # Find chunks mentioning taser, stun, electrical damage
        relevant_chunks = []

        for i, doc in enumerate(documents):
            doc_lower = doc.lower()

            # Keywords related to the query
            taser_keywords = ['taser', 'stun', 'electrical', 'electricity', 'shock']
            damage_keywords = ['damage', 'resist', 'soak', 'dice pool', 'body', 'armor']

            has_taser = any(keyword in doc_lower for keyword in taser_keywords)
            has_damage = any(keyword in doc_lower for keyword in damage_keywords)

            if has_taser or (has_damage and any(kw in doc_lower for kw in ['electric', 'stun'])):
                chunk_info = {
                    "chunk_index": i,
                    "classification": metadatas[i].get('primary_section', 'Unknown') if i < len(metadatas) else 'Unknown',
                    "word_count": len(doc.split()),
                    "has_explicit_rule": 'dice pool' in doc_lower and 'resist' in doc_lower,
                    "is_example": any(ex in doc_lower for ex in ['example', 'e.g.', 'for instance']),
                    "preview": doc[:200] + "..." if len(doc) > 200 else doc
                }
                relevant_chunks.append(chunk_info)

        return {
            "total_relevant_chunks": len(relevant_chunks),
            "chunks_with_explicit_rules": sum(1 for c in relevant_chunks if c['has_explicit_rule']),
            "chunks_with_examples_only": sum(
                1 for c in relevant_chunks if c['is_example'] and not c['has_explicit_rule']),
            "classification_distribution": dict(Counter(c['classification'] for c in relevant_chunks)),
            "sample_chunks": relevant_chunks[:3]  # First 3 for review
        }

    def _generate_recommendations(self, analysis):
        """Generate specific recommendations based on analysis."""
        recommendations = []

        # Size issues
        size_stats = analysis['size_analysis']['word_count_stats']
        if size_stats['std_dev'] > size_stats['mean'] * 0.8:
            recommendations.append("HIGH PRIORITY: Chunk sizes are highly inconsistent - implement adaptive chunking")

        # Classification issues
        section_dom = analysis['classification_analysis']['section_dominance']
        if section_dom['top_3_percentage'] > 80:
            recommendations.append(
                "HIGH PRIORITY: Classification is over-concentrated in few categories - improve multi-label classification")

        # Content quality issues
        quality = analysis['content_quality']
        if quality['example_vs_rules']['examples_only'] > quality['example_vs_rules']['rules_only']:
            recommendations.append(
                "MEDIUM PRIORITY: Too many example-only chunks - separate examples from rules with metadata flags")

        # Semantic boundary issues
        boundaries = analysis['semantic_boundaries']
        if boundaries['split_mid_sentence'] > analysis['total_chunks'] * 0.1:
            recommendations.append(
                "HIGH PRIORITY: >10% of chunks split mid-sentence - implement sentence-aware chunking")

        # Specific query issues
        specific = analysis['specific_issues']
        if specific['chunks_with_examples_only'] > specific['chunks_with_explicit_rules']:
            recommendations.append(
                "CRITICAL: Query-relevant chunks are mostly examples, not explicit rules - implement rule vs example tagging")

        return recommendations

    def print_analysis(self, analysis):
        """Print formatted analysis results."""

        print("\n" + "=" * 80)
        print("📊 SHADOWRUN RAG CHUNK QUALITY ANALYSIS")
        print("=" * 80)

        print(f"\n📈 OVERVIEW:")
        print(f"   Total chunks: {analysis['total_chunks']:,}")

        print(f"\n📏 CHUNK SIZES:")
        size_stats = analysis['size_analysis']['word_count_stats']
        print(
            f"   Words: {size_stats['min']}-{size_stats['max']} (avg: {size_stats['mean']:.0f}, median: {size_stats['median']:.0f})")
        print(f"   Size distribution:")
        for size_range, count in analysis['size_analysis']['size_distribution'].items():
            percentage = count / analysis['total_chunks'] * 100
            print(f"     {size_range}: {count:,} ({percentage:.1f}%)")

        print(f"\n🏷️  CLASSIFICATIONS:")
        sections = analysis['classification_analysis']['primary_sections']
        print(f"   Unique sections: {len(sections)}")
        for section, count in sorted(sections.items(), key=lambda x: x[1], reverse=True)[:5]:
            percentage = count / analysis['total_chunks'] * 100
            print(f"     {section}: {count:,} ({percentage:.1f}%)")

        dom = analysis['classification_analysis']['section_dominance']
        print(f"   Top 3 sections: {dom['top_3_percentage']:.1f}% of all chunks")

        print(f"\n🔍 CONTENT QUALITY:")
        quality = analysis['content_quality']
        print(f"   Empty chunks: {quality['empty_chunks']:,}")
        print(f"   Header-only chunks: {quality['header_only_chunks']:,}")
        print(f"   Incomplete sentences: {quality['incomplete_sentences']:,}")

        rules_info = quality['example_vs_rules']
        print(f"   Examples only: {rules_info['examples_only']:,} ({rules_info['example_percentage']:.1f}%)")
        print(f"   Rules only: {rules_info['rules_only']:,}")
        print(f"   Mixed/unclear: {rules_info['unclear_or_mixed']:,}")

        print(f"\n🧩 SEMANTIC BOUNDARIES:")
        boundaries = analysis['semantic_boundaries']
        print(f"   Mid-sentence splits: {boundaries['split_mid_sentence']:,}")
        print(f"   Mid-table splits: {boundaries['split_mid_table']:,}")
        print(f"   Orphaned headers: {boundaries['orphaned_headers']:,}")
        print(f"   Missing context: {boundaries['missing_context']:,}")

        print(f"\n⚡ SPECIFIC ANALYSIS (Taser/Stun Query):")
        specific = analysis['specific_issues']
        print(f"   Relevant chunks found: {specific['total_relevant_chunks']:,}")
        print(f"   With explicit rules: {specific['chunks_with_explicit_rules']:,}")
        print(f"   Examples only: {specific['chunks_with_examples_only']:,}")
        print(f"   Classifications: {specific['classification_distribution']}")

        if specific['sample_chunks']:
            print(f"\n   Sample relevant chunks:")
            for i, chunk in enumerate(specific['sample_chunks']):
                print(f"     {i + 1}. [{chunk['classification']}] {chunk['word_count']} words")
                print(
                    f"        Rule: {'Yes' if chunk['has_explicit_rule'] else 'No'} | Example: {'Yes' if chunk['is_example'] else 'No'}")
                print(f"        Preview: {chunk['preview'][:100]}...")

        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"   {i}. {rec}")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    analyzer = ChunkAnalyzer()
    analysis = analyzer.analyze_chunks()

    if analysis:
        analyzer.print_analysis(analysis)

        # Save detailed analysis to file
        output_file = Path("chunk_analysis_results.json")
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\n💾 Detailed analysis saved to: {output_file}")
    else:
        print("❌ Analysis failed - check ChromaDB connection")