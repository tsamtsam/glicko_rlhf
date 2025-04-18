# Cell 1: Import Libraries
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from typing import List, Dict, Any, Tuple, Optional, Union
from tqdm.auto import tqdm


class RobertaRewardModel:
    """
    A wrapper class for a fine-tuned RoBERTa model that predicts rewards based on
    question-answer pairs. This model can be used as a reward function for GRPO training.
    """
    
    def __init__(
        self,
        model_path: str = "./roberta-reward-model",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 512,
        batch_size: int = 16,
    ):
        """
        Initialize the reward model with a fine-tuned RoBERTa model.
        
        Args:
            model_path: Path to the saved RoBERTa model
            device: Device to run the model on ('cuda' or 'cpu')
            max_length: Maximum token length for inputs
            batch_size: Batch size for processing multiple examples
        """
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Load tokenizer and model from the fine-tuned checkpoint
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        print(f"Loaded reward model from {model_path}")
        print(f"Running on device: {self.device}")
    
    def get_reward(
        self, 
        question: str, 
        answer: str
    ) -> float:
        """
        Get the reward for a single question-answer pair.
        
        Args:
            question: The question text
            answer: The answer text
            
        Returns:
            A scalar reward value
        """
        text = f"Question: {question} Answer: {answer}"
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            reward = outputs.logits.item()
        
        return reward
    
    def batch_get_rewards(
        self, 
        questions: List[str], 
        answers: List[str]
    ) -> List[float]:
        """
        Get rewards for batches of question-answer pairs.
        
        Args:
            questions: List of question texts
            answers: List of answer texts
            
        Returns:
            List of reward values
        """
        assert len(questions) == len(answers), "Number of questions and answers must match"
        
        rewards = []
        
        # Process in batches
        for i in range(0, len(questions), self.batch_size):
            batch_questions = questions[i:i+self.batch_size]
            batch_answers = answers[i:i+self.batch_size]
            
            # Create texts for the batch
            texts = [f"Question: {q} Answer: {a}" for q, a in zip(batch_questions, batch_answers)]
            
            # Tokenize
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_rewards = outputs.logits.flatten().cpu().numpy().tolist()
            
            rewards.extend(batch_rewards)
        
        return rewards
        
# Cell 3: GRPORewardFunction Class
class GRPORewardFunction:
    """
    A reward function wrapper for GRPO training that uses the RoBERTa reward model.
    This class provides the interface expected by GRPO training loops.
    """
    
    def __init__(
        self,
        reward_model: RobertaRewardModel,
        reward_scaling: float = 1.0,
        reward_clip_min: Optional[float] = None,
        reward_clip_max: Optional[float] = None,
        normalize_rewards: bool = True
    ):
        """
        Initialize the GRPO reward function with a reward model.
        
        Args:
            reward_model: Instance of the RobertaRewardModel
            reward_scaling: Factor to scale rewards by
            reward_clip_min: Minimum reward value after clipping (None for no clipping)
            reward_clip_max: Maximum reward value after clipping (None for no clipping)
            normalize_rewards: Whether to normalize rewards to mean=0, std=1
        """
        self.reward_model = reward_model
        self.reward_scaling = reward_scaling
        self.reward_clip_min = reward_clip_min
        self.reward_clip_max = reward_clip_max
        self.normalize_rewards = normalize_rewards
        
        # Stats for reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
    
    def __call__(
        self, 
        batch: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Calculate rewards for a batch of LLM responses during GRPO training.
        
        Args:
            batch: A dictionary containing:
                - 'input_texts': List of input prompts/questions
                - 'output_texts': List of LLM-generated outputs/answers
                
        Returns:
            A tensor of reward values for each item in the batch
        """
        questions = batch["input_texts"]
        answers = batch["output_texts"]
        
        # Get raw rewards from the model
        rewards = self.reward_model.batch_get_rewards(questions, answers)
        
        # Update running statistics for normalization
        if self.normalize_rewards:
            if self.reward_count == 0:
                self.reward_mean = np.mean(rewards)
                self.reward_std = np.std(rewards) + 1e-8  # Avoid division by zero
            else:
                # Update running mean and std
                new_count = self.reward_count + len(rewards)
                new_mean = (self.reward_mean * self.reward_count + np.sum(rewards)) / new_count
                new_std = np.sqrt(
                    (self.reward_count * (self.reward_std**2 + (self.reward_mean - new_mean)**2) + 
                     np.sum((np.array(rewards) - new_mean)**2)) / new_count
                ) + 1e-8  # Avoid division by zero
                
                self.reward_mean = new_mean
                self.reward_std = new_std
                self.reward_count = new_count
            
            # Normalize rewards
            rewards = [(r - self.reward_mean) / self.reward_std for r in rewards]
        
        # Apply scaling
        rewards = [r * self.reward_scaling for r in rewards]
        
        # Apply optional clipping
        if self.reward_clip_min is not None or self.reward_clip_max is not None:
            clip_min = float('-inf') if self.reward_clip_min is None else self.reward_clip_min
            clip_max = float('inf') if self.reward_clip_max is None else self.reward_clip_max
            rewards = [max(clip_min, min(clip_max, r)) for r in rewards]
        
        return torch.tensor(rewards)