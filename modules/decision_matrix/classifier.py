"""
Decision Matrix Classifier.

Simple Decision Matrix Classification Algorithm inspired by Random Forest.
Uses voting system with weighted impact and feature importance.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class DecisionMatrixClassifier:
    """
    Simple Decision Matrix Classification Algorithm.
    
    Inspired by Random Forest voting system from Document1.pdf.
    
    Architecture:
    - Node 1: ATC vote (0 or 1)
    - Node 2: Range Oscillator vote (0 or 1)
    - Node 3: SPC vote (0 or 1) [optional]
    - Cumulative Vote: Weighted combination of all votes
    """
    
    indicators: List[str] = field(default_factory=lambda: ['atc', 'oscillator'])
    node_votes: Dict[str, int] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    independent_accuracy: Dict[str, float] = field(default_factory=dict)
    weighted_impact: Dict[str, float] = field(default_factory=dict)
    signal_strengths: Dict[str, float] = field(default_factory=dict)
    
    def add_node_vote(
        self,
        indicator: str,
        vote: int,
        signal_strength: float = 0.5,
        accuracy: Optional[float] = None,
    ) -> None:
        """
        Add vote from an indicator node.
        
        Args:
            indicator: Indicator name ('atc', 'oscillator', 'spc')
            vote: Vote value (0 or 1)
            signal_strength: Signal strength (0.0 to 1.0) - used for feature importance
            accuracy: Independent accuracy (0.0 to 1.0) - optional, will use signal_strength if not provided
        """
        self.node_votes[indicator] = vote
        self.signal_strengths[indicator] = signal_strength
        
        # Feature importance based on signal strength
        self.feature_importance[indicator] = signal_strength
        
        # Independent accuracy
        if accuracy is not None:
            self.independent_accuracy[indicator] = accuracy
        else:
            # Use signal strength as proxy for accuracy
            self.independent_accuracy[indicator] = signal_strength
    
    def calculate_weighted_impact(self) -> None:
        """
        Calculate weighted impact for each indicator.
        
        Weighted impact = how much each indicator contributes to the voting scheme.
        Should be balanced (not let one indicator dominate >30-40%).
        """
        # Calculate total importance
        total_importance = sum(self.feature_importance.values())
        
        if total_importance == 0:
            # Equal weights if no importance data
            equal_weight = 1.0 / len(self.indicators)
            for indicator in self.indicators:
                self.weighted_impact[indicator] = equal_weight
        else:
            # Weighted by feature importance
            for indicator in self.indicators:
                importance = self.feature_importance.get(indicator, 0.0)
                self.weighted_impact[indicator] = importance / total_importance
            
            # Check for over-representation (>40%)
            max_weight = max(self.weighted_impact.values()) if self.weighted_impact else 0.0
            if max_weight > 0.4:
                # Normalize to prevent over-representation
                # Redistribute weights so max is 40%
                scale_factor = 0.4 / max_weight
                for indicator in self.indicators:
                    self.weighted_impact[indicator] *= scale_factor
                
                # Redistribute remaining weight equally
                remaining = 1.0 - sum(self.weighted_impact.values())
                if remaining > 0:
                    equal_addition = remaining / len(self.indicators)
                    for indicator in self.indicators:
                        self.weighted_impact[indicator] += equal_addition
    
    def calculate_cumulative_vote(
        self,
        threshold: float = 0.5,
        min_votes: int = 2,
    ) -> Tuple[int, float, Dict[str, Dict]]:
        """
        Calculate cumulative vote from all nodes.
        
        Args:
            threshold: Minimum weighted score for positive vote (default: 0.5)
            min_votes: Minimum number of indicators that must vote positive (default: 2)
        
        Returns:
            Tuple of:
            - cumulative_vote: 1 if weighted score >= threshold, 0 otherwise
            - weighted_score: Calculated weighted score (0.0 to 1.0)
            - voting_breakdown: Dictionary with individual votes and weights
        """
        # Calculate weighted score
        weighted_score = 0.0
        voting_breakdown = {}
        positive_votes = 0
        
        for indicator in self.indicators:
            vote = self.node_votes.get(indicator, 0)
            weight = self.weighted_impact.get(indicator, 1.0 / len(self.indicators))
            contribution = vote * weight
            weighted_score += contribution
            
            voting_breakdown[indicator] = {
                'vote': vote,
                'weight': weight,
                'contribution': contribution,
            }
            
            if vote == 1:
                positive_votes += 1
        
        # Check minimum votes requirement
        if positive_votes < min_votes:
            return (0, weighted_score, voting_breakdown)
        
        # Final vote based on threshold
        cumulative_vote = 1 if weighted_score >= threshold else 0
        
        return (cumulative_vote, weighted_score, voting_breakdown)
    
    def get_metadata(self) -> Dict:
        """Get all metadata for display."""
        return {
            'node_votes': self.node_votes.copy(),
            'feature_importance': self.feature_importance.copy(),
            'independent_accuracy': self.independent_accuracy.copy(),
            'weighted_impact': self.weighted_impact.copy(),
            'signal_strengths': self.signal_strengths.copy(),
        }
    
    def reset(self) -> None:
        """Reset classifier for next symbol."""
        self.node_votes.clear()
        self.feature_importance.clear()
        self.independent_accuracy.clear()
        self.weighted_impact.clear()
        self.signal_strengths.clear()


__all__ = ["DecisionMatrixClassifier"]

