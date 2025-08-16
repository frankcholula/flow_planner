import minari


def create_mixed_carracing_dataset():
    EXPERT_ID = "Box2D/CarRacing-v3/expert-v0"
    SIMPLE_ID = "Box2D/CarRacing-v3/simple-v0"
    COMBINED_ID = "Box2D/CarRacing-v3/mixed-v0"

    local_datasets = minari.list_local_datasets()
    assert EXPERT_ID in local_datasets, f"Dataset `{EXPERT_ID}` not found."
    assert SIMPLE_ID in local_datasets, f"Dataset `{SIMPLE_ID}` not found."
    expert_dataset = minari.load_dataset(EXPERT_ID)
    simple_dataset = minari.load_dataset(SIMPLE_ID)
    print(
        f"Combining {len(expert_dataset.episode_indices)} expert and {len(simple_dataset.episode_indices)} simple episodes..."
    )
    combined_dataset = minari.combine_datasets(
        datasets_to_combine=[expert_dataset, simple_dataset],
        new_dataset_id=COMBINED_ID,
    )
    print(
        f"Successfully created '{COMBINED_ID}' with {len(combined_dataset.episode_indices)} total episodes."
    )


if __name__ == "__main__":
    create_mixed_carracing_dataset()
