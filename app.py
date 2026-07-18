import streamlit as st
from chains.email_generator import generate_email

st.set_page_config(
    page_title="AI Cold Email Generator",
    page_icon="📧",
    layout="centered"
)

st.title("📧 AI Cold Email Generator")

st.write("Paste the Job Description below and click Generate Email.")

job_description = st.text_area(
    "Job Description",
    height=250,
    placeholder="Paste Job Description Here..."
)

if st.button("Generate Email"):

    if job_description.strip() == "":
        st.error("Please enter a Job Description.")
    else:

        with st.spinner("Generating Cold Email..."):

            try:
                email = generate_email(job_description)

                st.success("Email Generated Successfully!")

                st.subheader("Generated Email")

                st.text_area(
                    "",
                    value=email,
                    height=400
                )

            except Exception as e:
                st.error(str(e))