#%%
import sys
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

# Add the parent directory to Python path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.insert(0, project_root)

# Now import the model - use relative import
# Change this line in src/components/main.py:
# from src.components.model import LinUCB

# To this (relative import):
from ..Components.model import LinUCB

class FCPSDataLoader:
    """Load and prepare FCPS data for training with correct paths"""
    
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = data_dir
        self.action_matrix = None
        self.feature_matrix = None
        self.sales_data = None
        self.health_scores = None
        self.item_mapping = None
        
    def load_all_data(self) -> bool:
        """Load all FCPS data files with correct paths"""
        print("📊 Loading FCPS data files...")
        
        try:
            # Define correct paths based on your project structure
            data_path = os.path.join(os.path.dirname(__file__), self.data_dir)
            
            # 1. Load Action Matrix
            action_path = os.path.join(data_path, "action_matrix.csv")
            if os.path.exists(action_path):
                self.action_matrix = pd.read_csv(action_path)
                print(f"   ✅ Action Matrix: {self.action_matrix.shape}")
            else:
                print(f"   ❌ Action matrix not found at {action_path}")
                return False
            
            # 2. Load Feature Matrix
            feature_path = os.path.join(data_path, "feature_matrix.csv")
            if os.path.exists(feature_path):
                self.feature_matrix = pd.read_csv(feature_path)
                print(f"   ✅ Feature Matrix: {self.feature_matrix.shape}")
            else:
                print(f"   ❌ Feature matrix not found at {feature_path}")
                return False
            
            # 3. Load Sales Data
            sales_path = os.path.join(data_path, "sales.csv")
            if os.path.exists(sales_path):
                self.sales_data = pd.read_csv(sales_path)
                print(f"   ✅ Sales Data: {len(self.sales_data)} records")
            else:
                print(f"   ❌ Sales data not found at {sales_path}")
                return False
            
            # 4. Load or Create Item Mapping
            mapping_path = os.path.join(data_path, "item_mapping.csv")
            if os.path.exists(mapping_path):
                mapping_df = pd.read_csv(mapping_path)
                self.item_mapping = dict(zip(mapping_df['item'], mapping_df['item_idx']))
                print(f"   ✅ Item Mapping: {len(self.item_mapping)} items")
            else:
                print("   ⚠️  Item mapping not found, creating from sales data...")
                self._create_item_mapping()
                # Save the mapping for future use
                mapping_df = pd.DataFrame({
                    'item': list(self.item_mapping.keys()),
                    'item_idx': list(self.item_mapping.values())
                })
                mapping_df.to_csv(mapping_path, index=False)
                print(f"   ✅ Created and saved item mapping with {len(self.item_mapping)} items")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_item_mapping(self):
        """Create item mapping from sales data"""
        unique_items = sorted(self.sales_data['description'].astype(str).unique())
        self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
    
    def prepare_training_data(self, max_time_slots: int = 50) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        """Prepare data for LinUCB training"""
        print("\n🎯 Preparing training data...")
        
        if self.feature_matrix is None or self.action_matrix is None:
            raise ValueError("Data not loaded. Call load_all_data() first.")
        
        contexts_list = []
        available_actions_list = []
        
        # Get unique time slots from feature matrix
        time_slots = sorted(self.feature_matrix['time_slot_id'].unique())
        time_slots = time_slots[:max_time_slots]  # Limit for testing
        
        print(f"   Processing {len(time_slots)} time slots...")
        
        for t in time_slots:
            # Get features for this time slot
            slot_features = self.feature_matrix[self.feature_matrix['time_slot_id'] == t]
            if len(slot_features) > 0:
                # Use nutritional features (skip meta columns)
                feature_cols = [col for col in slot_features.columns 
                               if col not in ['time_slot_id', 'item', 'item_idx']]
                if len(feature_cols) > 0:
                    context = slot_features.iloc[0][feature_cols].values.astype(float)
                    contexts_list.append(context)
            
            # Get available actions for this time slot
            slot_actions = self.action_matrix[self.action_matrix['time_slot_id'] == t]
            if len(slot_actions) > 0:
                # Skip time_slot_id column, get all item columns
                action_cols = [col for col in slot_actions.columns if col != 'time_slot_id']
                available_actions = slot_actions.iloc[0][action_cols].values.astype(int)
                available_actions_list.append(available_actions)
        
        print(f"   ✅ Prepared {len(contexts_list)} training sequences")
        print(f"   ✅ Context dimension: {contexts_list[0].shape[0] if contexts_list else 0}")
        
        return contexts_list, available_actions_list, time_slots
    
    def extract_sales_and_health_data(self) -> Tuple[List[np.ndarray], Dict[int, float]]:
        """Extract actual sales counts and health scores"""
        print("\n📈 Extracting sales and health data...")
        
        if self.sales_data is None:
            raise ValueError("Sales data not loaded.")
        
        n_items = len(self.item_mapping)
        
        # Create sales matrix: time_slot_id x item_idx -> sales_count
        unique_time_slots = sorted(self.sales_data['time_slot_id'].unique())
        
        sales_data_list = []
        health_scores_dict = {}
        
        # Initialize health scores with defaults
        for item_idx in range(n_items):
            health_scores_dict[item_idx] = 3.5  # Default health score
        
        # Extract actual health scores from data
        if 'HealthScore' in self.sales_data.columns:
            health_data = self.sales_data[['description', 'HealthScore']].drop_duplicates()
            for _, row in health_data.iterrows():
                item_desc = str(row['description'])
                if item_desc in self.item_mapping:
                    item_idx = self.item_mapping[item_desc]
                    health_scores_dict[item_idx] = float(row['HealthScore'])
        
        # Create sales vectors for each time slot
        for time_slot in unique_time_slots:
            slot_data = self.sales_data[self.sales_data['time_slot_id'] == time_slot]
            sales_vector = np.zeros(n_items)
            
            for _, row in slot_data.iterrows():
                item_desc = str(row['description'])
                if item_desc in self.item_mapping:
                    item_idx = self.item_mapping[item_desc]
                    sales_count = float(row['total']) if 'total' in row.columns else 0.0
                    sales_vector[item_idx] = sales_count
            
            sales_data_list.append(sales_vector)
        
        # Print statistics
        actual_health_scores = [score for score in health_scores_dict.values() if score != 3.5]
        if actual_health_scores:
            print(f"   ✅ Health scores range: {min(actual_health_scores):.1f}-{max(actual_health_scores):.1f}")
        print(f"   ✅ Extracted sales data for {len(sales_data_list)} time slots")
        
        return sales_data_list, health_scores_dict

def train_fcps_model():
    """Main training function with proper paths"""
    print("🍎 FCPS School Meal Recommendation System")
    print("=" * 50)
    
    # Initialize data loader with correct data directory
    loader = FCPSDataLoader("../data")
    
    try:
        # Step 1: Load all data
        if not loader.load_all_data():
            print("❌ Failed to load data. Please check file paths.")
            return None
        
        # Step 2: Prepare training data
        contexts_list, available_actions_list, time_slots = loader.prepare_training_data(max_time_slots=50)
        
        if len(contexts_list) == 0:
            print("❌ No training data prepared. Check your data files.")
            return None
        
        # Step 3: Extract sales and health data
        sales_data_list, health_scores = loader.extract_sales_and_health_data()
        
        # Step 4: Initialize model
        n_arms = len(loader.item_mapping)
        context_dim = contexts_list[0].shape[0]
        
        print(f"\n🧠 Model Configuration:")
        print(f"   Foods: {n_arms}, Features: {context_dim}")
        print(f"   Time slots: {len(contexts_list)}")
        
        model = LinUCB(n_arms=n_arms, context_dim=context_dim, alpha=1.0, lambda_reg=1.0)
        
        # Step 5: TRAIN THE MODEL!
        print("\n🚀 Starting Training...")
        rewards = model.train(
            contexts=contexts_list,
            available_actions_list=available_actions_list,
            sales_data=sales_data_list,
            health_scores=health_scores,
            lambda_param=0.3
        )
        
        # Step 6: Evaluate Results
        metrics = model.get_performance_metrics()
        print("\n📊 Training Results:")
        print(f"   Total decisions: {metrics['total_decisions']}")
        print(f"   Average reward: {metrics['average_reward']:.3f}")
        print(f"   Cumulative reward: {metrics['cumulative_reward']:.1f}")
        print(f"   Cumulative regret: {metrics['cumulative_regret']:.1f}")
        print(f"   Foods recommended: {metrics['foods_recommended']}/{n_arms}")
        
        # Step 7: Show Sample Recommendations
        print("\n💡 Sample Recommendations:")
        if contexts_list:
            sample_context = contexts_list[0]
            sample_available = available_actions_list[0]
            
            recommendations = model.recommend(sample_context, sample_available, top_k=5)
            print("   Top 5 recommended foods:")
            for i, (food_idx, score) in enumerate(recommendations):
                health = health_scores.get(food_idx, 0)
                # Find item name from mapping
                item_name = [k for k, v in loader.item_mapping.items() if v == food_idx]
                item_display = item_name[0] if item_name else f"Food #{food_idx}"
                print(f"   {i+1}. {item_display}: score={score:.2f}, health={health:.1f}")
        
        # Step 8: Save Model
        print("\n💾 Saving trained model...")
        models_dir = os.path.dirname(__file__)
        model_path = os.path.join(models_dir, "trained_fcps_model.json")
        model.save(model_path)
        
        print("\n🎉 SUCCESS! FCPS Model Training Complete!")
        print(f"   Model saved to: {model_path}")
        
        return model
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    trained_model = train_fcps_model()
# %%
