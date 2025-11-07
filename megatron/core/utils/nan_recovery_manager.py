"""NaN Recovery Manager for automatic NaN detection and recovery during training."""

from collections import defaultdict


class NaNRecoveryManager:
    """
    Manages NaN detection, tracking, and automatic hyperparameter adjustment.
    
    This manager provides:
    - Tracking of consecutive NaN occurrences
    - Recording of NaN hotspot parameters
    - Automatic hyperparameter adjustment based on NaN frequency
    - Checkpoint rollback recommendations
    """
    
    def __init__(self, args):
        """
        Initialize the NaN recovery manager.
        
        Args:
            args: Training arguments containing initial hyperparameters
        """
        self.consecutive_nan_count = 0
        self.total_nan_count = 0
        
        # Thresholds
        self.nan_threshold = getattr(args, 'nan_adjust_threshold', 3)
        self.checkpoint_rollback_threshold = getattr(args, 'nan_rollback_threshold', 5)
        
        # Hotspot tracking: parameter_name -> NaN occurrence count
        self.nan_hotspots = defaultdict(int)
        
        # Store initial hyperparameters for scaling
        self.initial_lr = args.lr
        self.initial_clip_grad = args.clip_grad
        
        # Gumbel temperature handling
        if hasattr(args, 'gumbel_temperature_range') and args.gumbel_temperature_range:
            self.initial_gumbel_temp_min = args.gumbel_temperature_range[1]
        else:
            self.initial_gumbel_temp_min = None
    
    def record_nan(self, param_names=None):
        """
        Record a NaN occurrence.
        
        Args:
            param_names: Optional list of parameter names that contained NaN
        """
        self.consecutive_nan_count += 1
        self.total_nan_count += 1
        
        # Track hotspot parameters
        if param_names:
            for param_name in param_names:
                self.nan_hotspots[param_name] += 1
    
    def reset_consecutive_count(self):
        """Reset the consecutive NaN counter (called after a successful step)."""
        self.consecutive_nan_count = 0
    
    def should_adjust_hyperparams(self):
        """
        Check if hyperparameters should be adjusted.
        
        Returns:
            bool: True if consecutive NaN count reaches threshold
        """
        return self.consecutive_nan_count >= self.nan_threshold
    
    def should_rollback_checkpoint(self):
        """
        Check if checkpoint rollback should be considered.
        
        Returns:
            bool: True if consecutive NaN count reaches rollback threshold
        """
        return self.consecutive_nan_count >= self.checkpoint_rollback_threshold
    
    def get_adjusted_params(self, args):
        """
        Get adjusted hyperparameters based on NaN frequency.
        
        Strategy:
        - Learning rate: Reduce by half for each threshold breach
        - Gradient clipping: Double for each threshold breach (capped at 2.0)
        - Gumbel temperature min: Increase to 2.5 for stability
        
        Args:
            args: Current training arguments
        
        Returns:
            dict: Dictionary with adjusted 'lr', 'clip_grad', and 'gumbel_temp_min'
        """
        # Calculate how many times we've crossed the threshold
        adjustment_count = self.consecutive_nan_count // self.nan_threshold
        
        # Learning rate: reduce by half for each adjustment
        lr_scale = 0.5 ** adjustment_count
        adjusted_lr = self.initial_lr * lr_scale
        
        # Gradient clipping: double for each adjustment, cap at 2.0
        adjusted_clip_grad = min(self.initial_clip_grad * (2 ** adjustment_count), 2.0)
        
        # Gumbel temperature: increase minimum to 2.5 for stability
        adjusted_gumbel_temp_min = None
        if self.initial_gumbel_temp_min is not None:
            adjusted_gumbel_temp_min = max(self.initial_gumbel_temp_min, 2.5)
        
        return {
            'lr': adjusted_lr,
            'clip_grad': adjusted_clip_grad,
            'gumbel_temp_min': adjusted_gumbel_temp_min
        }
    
    def get_top_hotspots(self, top_k=5):
        """
        Get the top-k parameters with most NaN occurrences.
        
        Args:
            top_k: Number of top hotspots to return
        
        Returns:
            list: List of (parameter_name, count) tuples sorted by count
        """
        if not self.nan_hotspots:
            return []
        
        sorted_hotspots = sorted(
            self.nan_hotspots.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_hotspots[:top_k]
    
    def get_statistics(self):
        """
        Get NaN statistics summary.
        
        Returns:
            dict: Dictionary with 'consecutive', 'total', and 'hotspot_count'
        """
        return {
            'consecutive': self.consecutive_nan_count,
            'total': self.total_nan_count,
            'hotspot_count': len(self.nan_hotspots)
        }

