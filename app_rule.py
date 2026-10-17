
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/recommend"

st.set_page_config(
    page_title="Market Basket Recommender",
    page_icon="🛒",
    layout="centered",
)

st.title("Market Basket Recommender")
st.write("Discover products associated with your current basket.")

products = ["bread", "milk", "eggs", "butter"]

selected_items = st.multiselect(
    "Products in your basket",
    options=products,
    placeholder="Select products",
)

if st.button("Get recommendations", type="primary"):
    if not selected_items:
        st.warning("Please select at least one product.")
    else:
        try:
            response = requests.post(
                API_URL,
                json={"items": selected_items},
                timeout=5,
            )
            response.raise_for_status()

            recommendations = response.json()["recommendations"]

            st.subheader("Recommended products")

            if recommendations:
                for product in recommendations:
                    st.success(product.capitalize())
            else:
                st.info("No recommendations found for this basket.")

        except requests.RequestException:
            st.error(
                "Could not connect to the API. "
                "Check that FastAPI is running."
            )
