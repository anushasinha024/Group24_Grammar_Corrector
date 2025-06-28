import os
import json
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import numpy as np
from nltk import pos_tag, word_tokenize, ngrams
from nltk.corpus import stopwords
import language_tool_python
import re

class GrammarCorrector:
    def __init__(self, dataset_path: str = "fce/json"):
        """
        Initialize the Grammar Corrector with hybrid approach:
        1. Rule-based checking using language-tool-python
        2. Statistical validation using FCE corpus
        3. NLTK for linguistic analysis
        """
        self.dataset_path = dataset_path
        
        # Initialize language tool
        self.lang_tool = language_tool_python.LanguageTool('en-US')
        
        # N-gram models from FCE corpus
        self.trigram_counts = Counter()
        self.bigram_counts = Counter()
        self.unigram_counts = Counter()
        self.total_words = 0
        
        # Error patterns from FCE
        self.error_patterns = defaultdict(list)  # POS pattern -> corrections
        self.error_stats = defaultdict(Counter)  # Error type statistics
        
        # Load NLTK resources
        try:
            import nltk
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('stopwords')
        except Exception as e:
            print(f"Warning: NLTK resource loading failed: {e}")
            
        # Train the model
        self.train()

    def apply_safe_corrections(self, text):
        """Apply corrections safely by sorting matches in reverse order"""
        matches = self.lang_tool.check(text)
        matches = sorted(matches, key=lambda m: m.offset, reverse=True)
        for match in matches:
            if match.replacements:
                start = match.offset
                end = start + match.errorLength
                replacement = match.replacements[0]
                text = text[:start] + replacement + text[end:]
        return text

    def _load_fce_file(self, filename: str) -> List[Dict]:
        """Load and parse an FCE JSON file."""
        filepath = os.path.join(self.dataset_path, filename)
        entries = []
        
        print(f"Loading file: {filepath}")
        print(f"File exists: {os.path.exists(filepath)}")
        
        if not os.path.exists(filepath):
            print(f"Warning: {filename} not found at {filepath}")
            return entries
            
        try:
            print(f"Opening file {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        entry = json.loads(line.strip())
                        if 'text' in entry and 'edits' in entry:
                            entries.append(entry)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding line {i+1}: {e}")
                        continue
                print(f"Loaded {len(entries)} entries")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            
        return entries

    def _extract_error_patterns(self, text: str, edits: List) -> List[Dict]:
        """Extract error patterns with POS context from FCE edits."""
        patterns = []
        
        # Tokenize and POS tag the entire text
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        
        for edit_group in edits:
            for edit in edit_group[1]:
                start, end = edit[0], edit[1]
                correction = edit[2]
                error_type = edit[3]
                
                # Get the original text
                original = text[start:end] if end > start else ""
                if not original or not correction:
                    continue
                
                # Get context tokens and POS tags
                orig_tokens = word_tokenize(original)
                orig_pos = pos_tag(orig_tokens)
                
                # Get correction tokens and POS tags
                corr_tokens = word_tokenize(correction)
                corr_pos = pos_tag(corr_tokens)
                
                # Create pattern
                pattern = {
                    'original': original,
                    'correction': correction,
                    'orig_pos': orig_pos,
                    'corr_pos': corr_pos,
                    'error_type': error_type,
                    'context': text[max(0, start-50):min(len(text), end+50)]
                }
                patterns.append(pattern)
                
                # Update error statistics
                self.error_stats[error_type]['total'] += 1
                pos_pattern = tuple(tag for _, tag in orig_pos)
                if 'pos_patterns' not in self.error_stats[error_type]:
                    self.error_stats[error_type]['pos_patterns'] = Counter()
                self.error_stats[error_type]['pos_patterns'][pos_pattern] += 1
                
        return patterns

    def _update_ngram_counts(self, text: str):
        """Update n-gram counts from text."""
        tokens = word_tokenize(text.lower())
        self.total_words += len(tokens)
        
        # Update counts
        self.unigram_counts.update(tokens)
        self.bigram_counts.update(ngrams(tokens, 2))
        self.trigram_counts.update(ngrams(tokens, 3))

    def train(self):
        """
        Train the model using FCE dataset:
        1. Build n-gram language models
        2. Extract error patterns
        3. Compute error statistics
        """
        print("Training model using FCE dataset...")
        
        # Process training data
        train_entries = self._load_fce_file('fce.train.json')
        print(f"Loaded {len(train_entries)} training examples")
        
        for entry in train_entries:
            # Update language model with corrected text
            self._update_ngram_counts(entry['text'])
            
            # Extract and store error patterns
            patterns = self._extract_error_patterns(entry['text'], entry['edits'])
            for pattern in patterns:
                pos_key = tuple(tag for _, tag in pattern['orig_pos'])
                self.error_patterns[pos_key].append(pattern)
        
        print(f"Built language model with {len(self.trigram_counts)} trigrams")
        print(f"Extracted patterns for {len(self.error_patterns)} POS sequences")
        
        # Print error type statistics
        print("\nError type distribution:")
        for error_type, stats in sorted(self.error_stats.items(), 
                                      key=lambda x: x[1]['total'], reverse=True):
            print(f"{error_type}: {stats['total']} occurrences")

    def _calculate_ngram_probability(self, sequence: List[str]) -> float:
        """Calculate probability of a sequence using trigram model with backoff."""
        if len(sequence) == 3:
            trigram = tuple(sequence)
            if trigram in self.trigram_counts:
                bigram_context = trigram[:2]
                if bigram_context in self.bigram_counts:
                    return self.trigram_counts[trigram] / self.bigram_counts[bigram_context]
        
        if len(sequence) >= 2:
            bigram = tuple(sequence[-2:])
            if bigram in self.bigram_counts:
                context = sequence[-2]
                if context in self.unigram_counts:
                    return self.bigram_counts[bigram] / self.unigram_counts[context]
        
        # Backoff to unigram with smoothing
        word = sequence[-1]
        return (self.unigram_counts[word] + 1) / (self.total_words + len(self.unigram_counts))

    def _validate_correction(self, original: str, correction: str, context: str) -> float:
        """
        Validate a proposed correction using the language model.
        Returns a confidence score between 0 and 1.
        """
        # Tokenize the context
        tokens = word_tokenize(context.lower())
        
        # Find position of original/correction
        try:
            idx = tokens.index(original.lower())
        except ValueError:
            return 0.5  # Default confidence if context doesn't match
            
        # Calculate probabilities using trigram context
        orig_prob = 1.0
        corr_prob = 1.0
        
        for i in range(max(0, idx-2), min(len(tokens), idx+3)):
            if i+2 < len(tokens):
                # Get trigram for original and correction
                orig_trigram = tokens[i:i+3]
                corr_trigram = orig_trigram.copy()
                
                # Replace the error word with correction
                if i <= idx < i+3:
                    corr_trigram[idx-i] = correction.lower()
                
                # Calculate probabilities
                orig_prob *= self._calculate_ngram_probability(orig_trigram)
                corr_prob *= self._calculate_ngram_probability(corr_trigram)
        
        # Convert to confidence score between 0 and 1
        if orig_prob == 0:
            return 1.0 if corr_prob > 0 else 0.5
        return min(1.0, corr_prob / orig_prob)

    def _check_subject_verb_agreement(self, tokens, pos_tags):
        """
        Enhanced subject-verb agreement checking with compound subjects
        and auxiliary verbs
        """
        errors = []
        for i, (token, pos) in enumerate(zip(tokens, pos_tags)):
            # Find main subject and verb pairs
            if pos[1].startswith('VB'):
                # Look for subject before the verb
                subject_pos = None
                subject_token = None
                compound_subject = False
                plural_subject = False
                
                # Search backwards for subject
                for j in range(i-1, -1, -1):
                    if pos_tags[j][1] in ['NN', 'NNS', 'PRP']:
                        subject_pos = pos_tags[j][1]
                        subject_token = tokens[j]
                        
                        # Check for compound subject (e.g., "John and Mary")
                        if j > 0 and tokens[j-1].lower() == 'and':
                            compound_subject = True
                            plural_subject = True
                        elif subject_pos == 'NNS' or subject_token.lower() in ['we', 'they']:
                            plural_subject = True
                        break
                
                if subject_pos:
                    # Check basic subject-verb agreement
                    if plural_subject and pos[1] == 'VBZ':
                        errors.append({
                            'start': i,
                            'end': i + 1,
                            'type': 'PLURAL_SUBJECT_VERB_AGREEMENT',
                            'correction': self._get_plural_verb_form(token)
                        })
                    elif not plural_subject and pos[1] == 'VBP':
                        errors.append({
                            'start': i,
                            'end': i + 1,
                            'type': 'SINGULAR_SUBJECT_VERB_AGREEMENT',
                            'correction': self._get_singular_verb_form(token)
                        })
                
                # Check for missing auxiliary verbs
                if pos[1] == 'VBG':  # Present participle
                    # Check if there's a "be" verb before
                    has_aux = False
                    for j in range(i-1, -1, -1):
                        if pos_tags[j][1].startswith('VB') and tokens[j].lower() in ['am', 'is', 'are', 'was', 'were']:
                            has_aux = True
                            break
                    
                    if not has_aux:
                        # Add appropriate form of "be"
                        if subject_token:
                            if subject_token.lower() == 'i':
                                aux = 'am'
                            elif plural_subject:
                                aux = 'are'
                            else:
                                aux = 'is'
                            errors.append({
                                'start': i,
                                'end': i,
                                'type': 'MISSING_AUXILIARY_BE',
                                'correction': f'{aux} {token}'
                            })
        
        return errors

    def _get_plural_verb_form(self, verb):
        """Convert a singular verb to its plural form"""
        if verb.endswith('s'):
            return verb[:-1]
        return verb

    def _get_singular_verb_form(self, verb):
        """Convert a plural verb to its singular form"""
        if not verb.endswith('s'):
            return verb + 's'
        return verb

    def _check_missing_auxiliary(self, tokens, pos_tags):
        """
        Check for missing auxiliary verbs in progressive
        and passive constructions
        """
        errors = []
        for i, (token, pos) in enumerate(zip(tokens, pos_tags)):
            # Check for present participle (-ing form)
            if pos[1] == 'VBG':
                # Look backwards for auxiliary 'be'
                has_be = False
                for j in range(i-1, max(-1, i-5), -1):
                    if j >= 0 and tokens[j].lower() in ['am', 'is', 'are', 'was', 'were']:
                        has_be = True
                        break
                
                if not has_be:
                    # Find subject to determine correct form of 'be'
                    subject = None
                    for j in range(i-1, max(-1, i-5), -1):
                        if j >= 0 and pos_tags[j][1] in ['NN', 'NNS', 'PRP']:
                            subject = tokens[j]
                            break
                    
                    if subject:
                        aux = self._get_be_form(subject)
                        errors.append({
                            'start': i,
                            'end': i,
                            'type': 'MISSING_PROGRESSIVE_BE',
                            'correction': f'{aux} {token}',
                            'confidence': 0.95
                        })
                    else:
                        # Default to 'am' if no subject found (common with 'I')
                        errors.append({
                            'start': i,
                            'end': i,
                            'type': 'MISSING_PROGRESSIVE_BE',
                            'correction': f'am {token}',
                            'confidence': 0.8
                        })
            
            # Check for past participle (passive voice)
            elif pos[1] == 'VBN':
                # Look backwards for auxiliary 'be' or 'have'
                has_aux = False
                for j in range(i-1, max(-1, i-5), -1):
                    if j >= 0 and (tokens[j].lower() in ['am', 'is', 'are', 'was', 'were'] or  # passive
                        tokens[j].lower() in ['have', 'has', 'had']):  # perfect
                        has_aux = True
                        break
                
                if not has_aux:
                    # Find subject to determine correct form
                    subject = None
                    for j in range(i-1, max(-1, i-5), -1):
                        if j >= 0 and pos_tags[j][1] in ['NN', 'NNS', 'PRP']:
                            subject = tokens[j]
                            break
                    
                    if subject:
                        # Default to passive (be + past participle)
                        aux = self._get_be_form(subject)
                        errors.append({
                            'start': i,
                            'end': i,
                            'type': 'MISSING_PASSIVE_BE',
                            'correction': f'{aux} {token}',
                            'confidence': 0.95
                        })
        
        return errors

    def _find_subject(self, tokens, pos_tags):
        """Find the subject for a verb"""
        for i in range(len(tokens)-1, -1, -1):
            if pos_tags[i][1] in ['NN', 'NNS', 'PRP']:
                return tokens[i]
        return None

    def _get_be_form(self, subject):
        """Get the correct form of 'be' based on the subject"""
        subject = subject.lower()
        if subject == 'i':
            return 'am'
        elif subject in ['we', 'you', 'they'] or subject.endswith('s'):  # plural
            return 'are'
        else:
            return 'is'

    def _check_article_number_agreement(self, tokens, pos_tags):
        """
        Check for article-noun agreement and count/mass noun distinctions
        """
        errors = []
        for i, (token, pos) in enumerate(zip(tokens, pos_tags)):
            # Check articles with nouns
            if pos[1] in ['DT', 'AT']:  # Determiners and articles
                if i < len(tokens) - 1:  # Check next token
                    next_token = tokens[i + 1]
                    next_pos = pos_tags[i + 1][1]
                    
                    # Check 'a' vs 'an'
                    if token.lower() == 'a' and next_token[0].lower() in 'aeiou':
                        errors.append({
                            'start': i,
                            'end': i + 1,
                            'type': 'A_AN_AGREEMENT',
                            'correction': 'an'
                        })
                    elif token.lower() == 'an' and next_token[0].lower() not in 'aeiou':
                        errors.append({
                            'start': i,
                            'end': i + 1,
                            'type': 'A_AN_AGREEMENT',
                            'correction': 'a'
                        })
                    
                    # Check plural nouns with singular articles
                    if token.lower() in ['a', 'an'] and next_pos == 'NNS':
                        errors.append({
                            'start': i,
                            'end': i + 2,
                            'type': 'ARTICLE_NUMBER_AGREEMENT',
                            'correction': f'the {next_token}'  # or remove article
                        })
            
            # Check quantifiers with nouns
            elif token.lower() in ['many', 'few', 'several', 'these', 'those']:
                if i < len(tokens) - 1:
                    next_token = tokens[i + 1]
                    next_pos = pos_tags[i + 1][1]
                    
                    # Should be followed by plural noun
                    if next_pos == 'NN':  # Singular noun after plural quantifier
                        plural_form = self._get_plural_noun_form(next_token)
                        errors.append({
                            'start': i + 1,
                            'end': i + 2,
                            'type': 'QUANTIFIER_NUMBER_AGREEMENT',
                            'correction': plural_form
                        })
            
            # Check numbers with nouns
            elif pos[1] == 'CD' and token != '1':  # Cardinal numbers except 1
                if i < len(tokens) - 1:
                    next_token = tokens[i + 1]
                    next_pos = pos_tags[i + 1][1]
                    
                    if next_pos == 'NN':  # Should be plural
                        plural_form = self._get_plural_noun_form(next_token)
                        errors.append({
                            'start': i + 1,
                            'end': i + 2,
                            'type': 'NUMBER_AGREEMENT',
                            'correction': plural_form
                        })
        
        return errors

    def _get_plural_noun_form(self, noun):
        """Convert a singular noun to its plural form"""
        # Basic English pluralization rules
        if noun.endswith('y'):
            return noun[:-1] + 'ies'
        elif noun.endswith(('s', 'sh', 'ch', 'x', 'z')):
            return noun + 'es'
        else:
            return noun + 's'

    def _apply_corrections(self, text: str, corrections: List[Dict]) -> str:
        """Apply corrections to text while handling overlapping changes"""
        if not corrections:
            return text
        
        # Sort corrections by position and confidence
        corrections.sort(key=lambda x: (x['offset'], -x['confidence']))
        
        # Group overlapping corrections and keep only the best ones
        filtered_corrections = []
        current_group = []
        
        for corr in corrections:
            if not current_group:
                current_group.append(corr)
                continue
            
            # Check if current correction overlaps with the group
            group_start = min(c['offset'] for c in current_group)
            group_end = max(c['offset'] + c['length'] for c in current_group)
            
            if corr['offset'] <= group_end:
                current_group.append(corr)
            else:
                # Process the current group
                if len(current_group) == 1:
                    filtered_corrections.append(current_group[0])
                else:
                    # For overlapping corrections, keep the one with highest confidence
                    best_corr = max(current_group, key=lambda x: x['confidence'])
                    filtered_corrections.append(best_corr)
                
                # Start new group
                current_group = [corr]
        
        # Process the last group
        if current_group:
            if len(current_group) == 1:
                filtered_corrections.append(current_group[0])
            else:
                best_corr = max(current_group, key=lambda x: x['confidence'])
                filtered_corrections.append(best_corr)
        
        # Sort corrections by position for application
        filtered_corrections.sort(key=lambda x: x['offset'], reverse=True)
        
        # Create a list of text segments
        segments = []
        last_end = 0
        
        # Sort corrections by start position
        filtered_corrections.sort(key=lambda x: x['offset'])
        
        # Build segments
        for corr in filtered_corrections:
            start = corr['offset']
            end = start + corr['length']
            
            # Add text before correction
            if start > last_end:
                segments.append(text[last_end:start])
            
            # Add correction
            segments.append(corr['correction'])
            last_end = end
        
        # Add remaining text
        if last_end < len(text):
            segments.append(text[last_end:])
        
        # Join segments
        return ''.join(segments)

    def correct_text(self, text: str) -> Dict:
        """
        Correct grammar in the input text using hybrid approach:
        1. Use LanguageTool for rule-based corrections
        2. Apply enhanced grammar checks
        3. Validate corrections using n-gram model
        """
        try:
            # Tokenize and POS tag the text
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # Get LanguageTool suggestions
            matches = self.lang_tool.check(text)
            
            # Collect all potential corrections
            corrections = []
            
            # Add LanguageTool corrections
            for match in matches:
                start = match.offset
                end = start + match.errorLength
                original = text[start:end]
                replacements = match.replacements
                
                if not replacements:
                    continue
                
                # Get context for validation
                context_start = max(0, start - 50)
                context_end = min(len(text), end + 50)
                context = text[context_start:context_end]
                
                # Find best correction
                best_correction = None
                best_confidence = 0
                
                for replacement in replacements:
                    confidence = self._validate_correction(original, replacement, context)
                    if confidence > best_confidence:
                        best_correction = replacement
                        best_confidence = confidence
                
                if best_correction and best_confidence > 0.5:
                    corrections.append({
                        'original': original,
                        'correction': best_correction,
                        'message': match.message,
                        'context': original,
                        'extended_context': context,
                        'offset': start,
                        'length': len(original),
                        'suggestions': replacements,
                        'category': match.ruleId,
                        'confidence': best_confidence
                    })
            
            # Apply corrections using the safe method
            corrected_text = self.apply_safe_corrections(text)
            
            return {
                'original': text,
                'corrected': corrected_text,
                'errors': corrections,
                'error_count': len(corrections)
            }
            
        except Exception as e:
            print(f"Error during correction: {e}")
            return {
                'original': text,
                'corrected': text,
                'errors': [],
                'error_count': 0
            }

    def evaluate(self, test_file: str = None) -> Dict:
        """
        Evaluate the model on test data.
        
        Args:
            test_file: Path to test file (defaults to fce.test.json)
            
        Returns:
            Dictionary containing evaluation metrics:
            - precision, recall, f1
            - per error type metrics
            - confusion matrix
        """
        if test_file is None:
            test_file = os.path.join(self.dataset_path, 'fce.test.json')
        elif os.path.isabs(test_file):
            # Keep absolute paths as is
            pass
        else:
            # For relative paths, don't join with dataset_path if it already includes it
            if not test_file.startswith(self.dataset_path):
                test_file = os.path.join(self.dataset_path, os.path.basename(test_file))
        
        print(f"Evaluating on {test_file}...")
        test_entries = self._load_fce_file(os.path.basename(test_file))
        if not test_entries:
            print("No test entries found!")
            return {}
            
        print(f"Loaded {len(test_entries)} test examples")
        
        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0
        per_type_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        # Track specific error types
        error_type_counts = defaultdict(int)
        correction_examples = defaultdict(list)
        
        for entry in test_entries:
            # Get ground truth corrections
            true_corrections = set()
            for edit_group in entry['edits']:
                for edit in edit_group[1]:
                    if edit[2] is not None:  # Ignore detection-only edits
                        start, end = edit[0], edit[1]
                        correction = edit[2]
                        error_type = edit[3]
                        original = entry['text'][start:end]
                        
                        true_corrections.add((start, end, correction, error_type))
                        error_type_counts[error_type] += 1
                        
                        # Store example corrections
                        if len(correction_examples[error_type]) < 3:  # Keep up to 3 examples per type
                            correction_examples[error_type].append({
                                'original': original,
                                'correction': correction,
                                'context': entry['text'][max(0, start-30):min(len(entry['text']), end+30)]
                            })
            
            # Get model predictions
            result = self.correct_text(entry['text'])
            pred_corrections = set()
            for error in result['errors']:
                pred_corrections.add((
                    error['offset'],
                    error['offset'] + error['length'],
                    error['correction'],
                    error['category']
                ))
            
            # Calculate metrics
            for true_corr in true_corrections:
                error_type = true_corr[3]
                if true_corr in pred_corrections:
                    total_true_positives += 1
                    per_type_metrics[error_type]['tp'] += 1
                else:
                    total_false_negatives += 1
                    per_type_metrics[error_type]['fn'] += 1
            
            for pred_corr in pred_corrections:
                error_type = pred_corr[3]
                if pred_corr not in true_corrections:
                    total_false_positives += 1
                    per_type_metrics[error_type]['fp'] += 1
        
        # Calculate overall metrics
        precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
        recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate per-type metrics
        type_metrics = {}
        for error_type, metrics in per_type_metrics.items():
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            
            type_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            type_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            type_f1 = 2 * type_precision * type_recall / (type_precision + type_recall) if (type_precision + type_recall) > 0 else 0
            
            type_metrics[error_type] = {
                'precision': type_precision,
                'recall': type_recall,
                'f1': type_f1,
                'support': tp + fn,
                'examples': correction_examples.get(error_type, [])
            }
        
        # Print detailed report
        print("\nEvaluation Results:")
        print(f"Total examples: {len(test_entries)}")
        print(f"True positives: {total_true_positives}")
        print(f"False positives: {total_false_positives}")
        print(f"False negatives: {total_false_negatives}")
        print(f"\nOverall Metrics:")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
        
        print("\nTop 5 Most Common Error Types:")
        for error_type, count in sorted(error_type_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            metrics = type_metrics[error_type]
            print(f"\n{error_type} (Count: {count})")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1: {metrics['f1']:.3f}")
            if metrics['examples']:
                print("  Example:")
                example = metrics['examples'][0]
                print(f"    Original: {example['original']}")
                print(f"    Correction: {example['correction']}")
        
        return {
            'overall': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'true_positives': total_true_positives,
                'false_positives': total_false_positives,
                'false_negatives': total_false_negatives,
                'total_examples': len(test_entries)
            },
            'per_type': type_metrics,
            'error_type_counts': dict(error_type_counts)
        } 