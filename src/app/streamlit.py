import requests
import streamlit as st
import openpyxl

# Define the API endpoint URL
API_URL = "http://localhost:8000/predict"

# Define a function to send a POST request to the API and get the response
def predict_sentiment(text):
    payload = {"text": text}
    response = requests.post(API_URL, json=payload)
    response.raise_for_status()
    return response.json()

def callback():
    st.session_state.button_clicked = True

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

# Define the Streamlit app
def main():
    st.title("Recognizing Hate Speech with Deep Learning")
    st.write("Enter some text and click the 'Analyze' button to get the sentiment analysis results.")
    
    # Add a text input field for the user to enter some text
    text = st.text_input("Enter some text here:")
    
    # Add a button to submit the text and get the sentiment analysis results
    if st.button("Analyze", on_click=callback) or st.session_state.button_clicked:
        if not text:
            st.warning("Please enter some text first.")
        else:
            label = predict_sentiment(text)

            with st.container():
                # Set the background color of the output box based on the predicted class
                if label['sentiment'] == 'Positive':
                    bg_color = '#C8E6C9'
                else:
                    bg_color = '#FFCDD2'

                # Define css styles for the output text
                css = f"""
                        <style>
                            .st-at {{
                                background-color: {bg_color};
                            }}
                        </style>
                    """
                st.write(css, unsafe_allow_html=True)
                st.info(f"Predicted sentiment label: {label['sentiment']}")

                # Ask the user if the sentiment prediction was correct or not
                with st.container():
                    st.write("Was the prediction correct?")
                    yes = st.button("Yes")
                    no = st.button("No")
                    
                    if yes:
                        st.session_state['yes_or_no'] = 'Yes'
                    if no:
                        st.session_state['yes_or_no'] = 'No'
                
    if 'yes_or_no' in st.session_state.keys():
        # Save the user feedback in an Excel file
        wb = openpyxl.load_workbook("src/app/user_feedback.xlsx")
        ws = wb.active
        
        if st.session_state['yes_or_no'] == 'Yes':
            real_label = label['sentiment']
        else:
            real_label = 'Positive' if label['sentiment'] == 'Negative' else 'Negative'

        ws.append([text, real_label])
        wb.save("src/app/user_feedback.xlsx")
        st.success("Thank you for your feedback!")
        del st.session_state['yes_or_no']

                
if __name__ == "__main__":
    main()
