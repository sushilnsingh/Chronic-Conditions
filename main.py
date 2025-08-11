# Entry point for the AI Pricing Tool

from data_generator import generate_deal_data
from pricing_model import train_model, recommend_price


def main() -> None:
    # Step 1: Generate synthetic data
    records = generate_deal_data(num_rows=1000)

    # Step 2: Train model
    model = train_model(records)

    # Step 3: Recommend pricing
    deal_size = 50000.0
    cost_basis = 30000.0
    recommendation = recommend_price(model, deal_size, cost_basis)

    print("Recommended Pricing Scenario:")
    print(recommendation)


if __name__ == "__main__":
    main()
