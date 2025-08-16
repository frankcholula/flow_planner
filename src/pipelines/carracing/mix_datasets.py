import minari

def create_mixed_carracing_dataset():
    EXPERT_ID = "Box2D/CarRacing-v3/expert-v0"
    SIMPLE_ID = "Box2D/CarRacing-v3/simple-v0"
    COMBINED_ID = "Box2D/CarRacing-v3/mixed-v0"
    NUM_EPISODES_EACH = 125

    local_datasets = minari.list_local_datasets()
    assert EXPERT_ID in local_datasets, f"Dataset `{EXPERT_ID}` not found."
    assert SIMPLE_ID in local_datasets, f"Dataset `{SIMPLE_ID}` not found."

    print("Creating temporary subsets...")
    expert_subset = minari.split_dataset(
        dataset_id=EXPERT_ID, sizes=(NUM_EPISODES_EACH,)
    )[0]
    simple_subset = minari.split_dataset(
        dataset_id=SIMPLE_ID, sizes=(NUM_EPISODES_EACH,)
    )[0]

    try:
        print(
            f"Combining {len(expert_subset.episode_indices)} expert and {len(simple_subset.episode_indices)} simple episodes..."
        )
        combined_dataset = minari.combine_datasets(
            datasets=[expert_subset, simple_subset],
            new_dataset_id=COMBINED_ID,
            code_permalink="https://github.com/frankcholula/flow_planner",
            author="Frank Lu",
            author_email="lu.phrank@gmail.com",
            description="A mix of 125 expert episodes and 125 random policy episodes.",
        )
        print(
            f"Successfully created '{COMBINED_ID}' with {len(combined_dataset.episode_indices)} total episodes."
        )

    finally:
        print("Cleaning up temporary subset datasets...")
        minari.delete_dataset(expert_subset.id)
        minari.delete_dataset(simple_subset.id)


if __name__ == "__main__":
    create_mixed_carracing_dataset()
