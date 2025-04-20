import pandas as pd
import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_submission(predictions_file, output_file):
    """Format the predictions into a submission file"""
    logger.info(f"Loading predictions from {predictions_file}")
    df = pd.read_csv(predictions_file)
    
    # Ensure we have the right columns
    if 'Student_IDs' not in df.columns:
        logger.error("Error: 'Student_IDs' column not found in predictions file")
        return
    
    if 'Occupancy' not in df.columns:
        logger.error("Error: 'Occupancy' column not found in predictions file")
        return
    
    # Create a new DataFrame for submission
    submission_df = pd.DataFrame({
        'Student_IDs': df['Student_IDs'],
        'Occupancy': df['Occupancy'].round().astype(int)  # Round to nearest integer
    })
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the submission file
    submission_df.to_csv(output_file, index=False)
    logger.info(f"Submission file saved to {output_file}")
    
    # Display some statistics about the submission
    logger.info("\nSubmission statistics:")
    logger.info(f"Number of predictions: {len(submission_df)}")
    logger.info(f"Occupancy range: {submission_df['Occupancy'].min()} to {submission_df['Occupancy'].max()}")
    logger.info(f"Mean occupancy: {submission_df['Occupancy'].mean():.2f}")
    logger.info(f"Median occupancy: {submission_df['Occupancy'].median()}")
    
    return submission_df

def main():
    parser = argparse.ArgumentParser(description='Format predictions into a submission file')
    parser.add_argument('--predictions', type=str, default='occupancy_prediction/predictions/predicted_occupancy.csv',
                        help='Path to the predictions file')
    parser.add_argument('--output', type=str, default='occupancy_prediction/predictions/submission.csv',
                        help='Path to save the submission file')
    args = parser.parse_args()
    
    create_submission(args.predictions, args.output)

if __name__ == '__main__':
    main() 