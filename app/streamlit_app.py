"""Demo entry point: names the two pages in Vietnamese (`make app`)."""

import streamlit as st

st.navigation([
    st.Page("main.py", title="Truy vết khách hàng", default=True),
    st.Page("pages/1_Mo_phong_online.py", title="Mô phỏng online", url_path="Mo_phong_online"),
]).run()
