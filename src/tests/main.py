# main.py - COMPLETE FCPS INTEGRATION
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Add project root to Python path
project_root = pathlib.Path(__file__).resolve().parents[0]
sys.path.append(str(project_root))

from src.components.model import LinUCB

class FCPSDataLoader:
    """Load and prepare FCPS data for training"""
    
    def __init__(self, data_dir="src/data"):
        self.data_dir = data_dir
        self.action_matrix = None
        self.feature_matrix = None
        self.time_slots_info = None
        self.sales_data = None
        self.health_scores = None
        
    def load_all_data(self):
        """Load all FCPS data files"""
        print("📊 Loading FCPS data files...")
        
        # 1. Load Action Matrix (food availability)
        action_path = os.path.join(self.data_dir, "action_matrix.csv")
        self.action_matrix = pd.read_csv(action_path)
        print(f"   ✅ Action Matrix: {self.action_matrix.shape}")
        
        # 2. Load Feature Matrix (nutritional features)
        feature_path = os.path.join(self.data_dir, "feature_matrix.csv")
        self.feature_matrix = pd.read_csv(feature_path)
        print(f"   ✅ Feature Matrix: {self.feature_matrix.shape}")
        
        # 3. Load Time Slots Info
        time_slots_path = os.path.join(self.data_dir, "time_slots_info.csv")
        self.time_slots_info = pd.read_csv(time_slots_path)
        print(f"   ✅ Time Slots: {len(self.time_slots_info)} periods")
        
        # 4. Load Sales Data (from original CSV)
        sales_path = os.path.join(self.data_dir, "sales.csv")
        sales_raw = pd.read_csv(sales_path)
        print(f"   ✅ Raw Sales: {len(sales_raw)} records")
        
        return True
    
    def prepare_training_data(self, max_time_slots=50):
        """Prepare data for LinUCB training"""
        print("\n🎯 Preparing training data...")
        
        # Extract contexts and available actions
        contexts_list = []
        available_actions_list = []
        
        # Get unique time slots
        time_slots = sorted(self.feature_matrix['time_slot_id'].unique())
        time_slots = time_slots[:max_time_slots]  # Limit for testing
        
        for t in time_slots:
            # Get features for this time slot
            slot_features = self.feature_matrix[self.feature_matrix['time_slot_id'] == t]
            if len(slot_features) > 0:
                # Use first food's features as context (they represent the time slot)
                context = slot_features.iloc[0, 3:].values  # Skip meta columns
                contexts_list.append(context)
            
            # Get available actions for this time slot
            slot_actions = self.action_matrix[self.action_matrix['time_slot_id'] == t]
            if len(slot_actions) > 0:
                available_actions = slot_actions.iloc[0, 1:].values  # Skip time_slot_id column
                available_actions_list.append(available_actions)
        
        print(f"   Prepared {len(contexts_list)} training sequences")
        return contexts_list, available_actions_list, time_slots
    
    def extract_sales_and_health_data(self):
        """Extract actual sales counts and health scores"""
        print("\n📈 Extracting sales and health data...")
        
        # Load original sales data to get actual sales counts
        sales_path = os.path.join(self.data_dir, "sales.csv")
        sales_df = pd.read_csv(sales_path)
        
        # Create sales matrix: time_slot_id x item_idx -> sales_count
        sales_matrix = {}
        health_scores_dict = {}
        
        # Group by time_slot_id and item
        for time_slot in sales_df['time_slot_id'].unique():
            slot_data = sales_df[sales_df['time_slot_id'] == time_slot]
            sales_matrix[time_slot] = {}
            
            for _, row in slot_data.iterrows():
                # Find item index (you'll need to map description to item_idx)
                item_desc = row['description']
                # This mapping should come from your item_mapping.csv
                # For now, we'll simulate
                item_idx = hash(item_desc) % 160  # Temporary - replace with actual mapping
                
                sales_count = row['total']  # Actual sales count
                health_score = row['HealthScore']  # Health score from data
                
                sales_matrix[time_slot][item_idx] = sales_count
                health_scores_dict[item_idx] = health_score
        
        print(f"   Extracted sales data for {len(sales_matrix)} time slots")
        print(f"   Health scores for {len(health_scores_dict)} items")
        
        return sales_matrix, health_scores_dict

def train_fcps_model():
    """Main training function"""
    print("🍎 FCPS School Meal Recommendation System")
    print("=" * 50)
    
    # Initialize data loader
    loader = FCPSDataLoader("src/data")
    
    try:
        # Step 1: Load all data
        loader.load_all_data()
        
        # Step 2: Prepare training data
        contexts_list, available_actions_list, time_slots = loader.prepare_training_data(max_time_slots=50)
        
        # Step 3: Extract sales and health data
        sales_matrix, health_scores = loader.extract_sales_and_health_data()
        
        # Step 4: Initialize model (160 foods, 19 nutritional features)
        n_arms = 160
        context_dim = 19  # Based on your feature matrix columns
        model = LinUCB(n_arms=n_arms, context_dim=context_dim, alpha=1.0)
        
        print(f"\n🧠 Model Initialized:")
        print(f"   Foods: {n_arms}, Features: {context_dim}")
        
        # Step 5: Convert sales data to training format
        sales_data_list = []
        for t in time_slots:
            if t in sales_matrix:
                # Create sales vector for all 160 items
                sales_vector = np.zeros(160)
                for item_idx, sales_count in sales_matrix[t].items():
                    if item_idx < 160:  # Ensure valid index
                        sales_vector[item_idx] = sales_count
                sales_data_list.append(sales_vector)
            else:
                # Fallback: random sales
                sales_data_list.append(np.random.poisson(30, 160))
        
        # Step 6: TRAIN THE MODEL!
        print("\n🚀 Starting Training...")
        model.train(
            contexts=contexts_list,
            available_actions_list=available_actions_list,
            sales_data=sales_data_list,
            health_scores=health_scores,
            lambda_param=0.3
        )
        
        # Step 7: Evaluate Results
        print("\n📊 Training Results:")
        print(f"   Total decisions: {len(model.rewards_list)}")
        print(f"   Average reward: {np.mean(model.rewards_list):.2f}")
        print(f"   Foods recommended: {np.sum(model.arm_counts > 0)}/160")
        
        # Step 8: Show Sample Recommendations
        print("\n💡 Sample Recommendations:")
        if contexts_list:
            sample_context = contexts_list[0]
            sample_available = available_actions_list[0]
            
            recommendations = model.recommend(sample_context, sample_available, top_k=5)
            print("   Top 5 recommended foods:")
            for i, (food_idx, score) in enumerate(recommendations):
                health = health_scores.get(food_idx, 0)
                print(f"   {i+1}. Food #{food_idx}: score={score:.2f}, health={health:.1f}")
        
        # Step 9: Save Model
        print("\n💾 Saving trained model...")
        os.makedirs("models", exist_ok=True)
        model.save("models/fcps_trained_model.json")
        
        print("\n🎉 SUCCESS! FCPS Model Training Complete!")
        print("   Next: Use this model for real cafeteria recommendations")
        
        return model
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    trained_model = train_fcps_model()