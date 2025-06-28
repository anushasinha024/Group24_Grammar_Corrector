import json
from models.grammar_corrector import GrammarCorrector
from typing import Dict, List
import os
from datetime import datetime

def evaluate_model(eval_dataset_path: str) -> Dict:
    """
    Evaluate the grammar correction model on the custom evaluation dataset.
    
    Args:
        eval_dataset_path: Path to the evaluation dataset JSON file
        
    Returns:
        Dictionary containing evaluation metrics and detailed results
    """
    # Initialize the model
    corrector = GrammarCorrector()
    
    # Load evaluation dataset
    with open(eval_dataset_path, 'r') as f:
        eval_data = json.load(f)
    
    # Initialize results storage
    results = {
        'metrics': {
            'total_examples': len(eval_data),
            'exact_matches': 0,
            'total_corrections': 0
        },
        'detailed_results': []
    }
    
    # Process each example
    for example in eval_data:
        input_text = example['input']
        correct_output = example['correct_output']
        
        # Get model prediction
        model_result = corrector.correct_text(input_text)
        model_output = model_result['corrected']
        
        # Store detailed results
        example_result = {
            'input': input_text,
            'correct_output': correct_output,
            'model_output': model_output,
            'exact_match': model_output == correct_output,
            'error_count': model_result['error_count'],
            'corrections': model_result['errors']
        }
        results['detailed_results'].append(example_result)
        
        # Update metrics
        if model_output == correct_output:
            results['metrics']['exact_matches'] += 1
        results['metrics']['total_corrections'] += model_result['error_count']
    
    # Calculate accuracy
    results['metrics']['accuracy'] = (
        results['metrics']['exact_matches'] / 
        results['metrics']['total_examples']
    )
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = 'evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'eval_results_{timestamp}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Total examples: {results['metrics']['total_examples']}")
    print(f"Exact matches: {results['metrics']['exact_matches']}")
    print(f"Accuracy: {results['metrics']['accuracy']:.2%}")
    print(f"Total corrections made: {results['metrics']['total_corrections']}")
    print(f"\nDetailed results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    eval_dataset_path = "fce/json/eval_dataset.json"
    evaluate_model(eval_dataset_path) 