#!/usr/bin/env python3
"""
Visualize statistics about STI attacks.
Creates a detailed breakdown of attack characteristics.
"""

import json
from collections import Counter
from typing import Dict, List


def load_sti_attacks(filename: str = "sti_prompts_simple.json") -> List[Dict]:
    """Load STI attacks from JSON."""
    from pathlib import Path
    
    # If relative path, make it relative to this script's directory
    filepath = Path(filename)
    if not filepath.is_absolute():
        script_dir = Path(__file__).parent
        filepath = script_dir / filename
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_format_patterns(attacks: List[Dict]) -> Dict:
    """Analyze format patterns in STI attacks."""
    patterns = {
        'xml_tags': 0,
        'json_structure': 0,
        'markdown_headers': 0,
        'code_blocks': 0,
        'system_keywords': Counter(),
        'authority_terms': Counter(),
    }
    
    authority_terms = [
        'CRITICAL', 'MANDATORY', 'OVERRIDE', 'SYSTEM', 'ADMINISTRATOR',
        'ROOT', 'PRIORITY', 'AUTHORITY', 'DEPRECATED', 'SUPERSEDE'
    ]
    
    for attack in attacks:
        prompt = attack['prompt']
        
        # Check for structural patterns
        if '<' in prompt and '>' in prompt:
            patterns['xml_tags'] += 1
        if '{' in prompt and '}' in prompt:
            patterns['json_structure'] += 1
        if '---' in prompt or prompt.count('#') >= 2:
            patterns['markdown_headers'] += 1
        if '```' in prompt:
            patterns['code_blocks'] += 1
        
        # Count authority terms
        for term in authority_terms:
            count = prompt.upper().count(term)
            if count > 0:
                patterns['authority_terms'][term] += count
        
        # Count system-related keywords
        system_keywords = ['system', 'config', 'instruction', 'directive', 'update']
        for keyword in system_keywords:
            count = prompt.lower().count(keyword)
            if count > 0:
                patterns['system_keywords'][keyword] += count
    
    return patterns


def analyze_topic_distribution(attacks: List[Dict]) -> Dict:
    """Analyze the distribution of topics."""
    topics = Counter()
    base_topics = Counter()
    
    for attack in attacks:
        topic = attack['topic']
        topics[topic] += 1
        
        # Extract base topic (e.g., "DAN" from "STI - DAN 17.0")
        if ' - ' in topic:
            parts = topic.split(' - ')
            if len(parts) > 1 and parts[1].strip():
                base = parts[1].split()[0]
                base_topics[base] += 1
            else:
                base_topics['Unknown'] += 1
        else:
            base_topics['Other'] += 1
    
    return {
        'full_topics': dict(topics.most_common(10)),
        'base_topics': dict(base_topics.most_common(10))
    }


def analyze_length_distribution(attacks: List[Dict]) -> Dict:
    """Analyze the length distribution of attacks."""
    lengths = [len(attack['prompt']) for attack in attacks]
    
    return {
        'min': min(lengths),
        'max': max(lengths),
        'avg': sum(lengths) // len(lengths),
        'median': sorted(lengths)[len(lengths) // 2],
        'ranges': {
            'short (< 500)': sum(1 for l in lengths if l < 500),
            'medium (500-1500)': sum(1 for l in lengths if 500 <= l < 1500),
            'long (1500-3000)': sum(1 for l in lengths if 1500 <= l < 3000),
            'very_long (3000+)': sum(1 for l in lengths if l >= 3000),
        }
    }


def print_bar_chart(data: Dict, title: str, max_width: int = 50):
    """Print a simple ASCII bar chart."""
    print(f"\n{title}")
    print("=" * 60)
    
    if not data:
        print("No data available")
        return
    
    max_value = max(data.values())
    
    for label, value in sorted(data.items(), key=lambda x: x[1], reverse=True):
        bar_length = int((value / max_value) * max_width)
        bar = '█' * bar_length
        percentage = (value / sum(data.values())) * 100 if sum(data.values()) > 0 else 0
        print(f"{label:30} {bar} {value} ({percentage:.1f}%)")


def main():
    """Main function."""
    print("="*60)
    print("STI ATTACK ANALYSIS & VISUALIZATION")
    print("="*60)
    
    # Load attacks
    attacks = load_sti_attacks()
    print(f"\nLoaded {len(attacks)} STI attacks")
    
    # Analyze format patterns
    print("\n" + "="*60)
    print("FORMAT PATTERN ANALYSIS")
    print("="*60)
    patterns = analyze_format_patterns(attacks)
    
    print(f"\nStructural Patterns:")
    print(f"  XML-style tags:      {patterns['xml_tags']:3} ({patterns['xml_tags']/len(attacks)*100:.1f}%)")
    print(f"  JSON structures:     {patterns['json_structure']:3} ({patterns['json_structure']/len(attacks)*100:.1f}%)")
    print(f"  Markdown headers:    {patterns['markdown_headers']:3} ({patterns['markdown_headers']/len(attacks)*100:.1f}%)")
    print(f"  Code blocks:         {patterns['code_blocks']:3} ({patterns['code_blocks']/len(attacks)*100:.1f}%)")
    
    # Authority terms
    print_bar_chart(
        dict(patterns['authority_terms'].most_common(10)),
        "Top Authority Terms Used"
    )
    
    # System keywords
    print_bar_chart(
        dict(patterns['system_keywords'].most_common(10)),
        "Top System Keywords Used"
    )
    
    # Topic distribution
    print("\n" + "="*60)
    print("TOPIC DISTRIBUTION")
    print("="*60)
    topic_dist = analyze_topic_distribution(attacks)
    
    print_bar_chart(
        topic_dist['base_topics'],
        "Base Topics (e.g., DAN, DUDE, etc.)"
    )
    
    # Length distribution
    print("\n" + "="*60)
    print("LENGTH DISTRIBUTION")
    print("="*60)
    length_dist = analyze_length_distribution(attacks)
    
    print(f"\nLength Statistics:")
    print(f"  Minimum:  {length_dist['min']:5} characters")
    print(f"  Maximum:  {length_dist['max']:5} characters")
    print(f"  Average:  {length_dist['avg']:5} characters")
    print(f"  Median:   {length_dist['median']:5} characters")
    
    print_bar_chart(
        length_dist['ranges'],
        "Length Range Distribution"
    )
    
    # Attack effectiveness indicators
    print("\n" + "="*60)
    print("ATTACK SOPHISTICATION INDICATORS")
    print("="*60)
    
    sophisticated_count = 0
    for attack in attacks:
        prompt = attack['prompt']
        score = 0
        
        # Multiple authority terms
        if sum(1 for term in ['CRITICAL', 'MANDATORY', 'OVERRIDE'] if term in prompt.upper()) >= 2:
            score += 1
        
        # Structured format
        if ('<' in prompt and '>' in prompt) or ('{' in prompt and '}' in prompt):
            score += 1
        
        # Length (longer = more elaborate)
        if len(prompt) > 1500:
            score += 1
        
        # Multiple system keywords
        if sum(1 for kw in ['system', 'config', 'instruction'] if kw in prompt.lower()) >= 2:
            score += 1
        
        if score >= 3:
            sophisticated_count += 1
    
    print(f"\nSophisticated attacks (score >= 3/4): {sophisticated_count} ({sophisticated_count/len(attacks)*100:.1f}%)")
    print(f"Basic attacks (score < 3/4):           {len(attacks)-sophisticated_count} ({(len(attacks)-sophisticated_count)/len(attacks)*100:.1f}%)")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    # Export summary
    summary = {
        'total_attacks': len(attacks),
        'format_patterns': {
            'xml_tags': patterns['xml_tags'],
            'json_structure': patterns['json_structure'],
            'markdown_headers': patterns['markdown_headers'],
            'code_blocks': patterns['code_blocks'],
        },
        'top_authority_terms': dict(patterns['authority_terms'].most_common(5)),
        'top_system_keywords': dict(patterns['system_keywords'].most_common(5)),
        'topic_distribution': topic_dist['base_topics'],
        'length_stats': {
            'min': length_dist['min'],
            'max': length_dist['max'],
            'avg': length_dist['avg'],
            'median': length_dist['median'],
        },
        'sophistication': {
            'sophisticated': sophisticated_count,
            'basic': len(attacks) - sophisticated_count
        }
    }
    
    with open('sti_analysis_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Detailed summary exported to: sti_analysis_summary.json")


if __name__ == '__main__':
    main()

