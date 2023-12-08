import streamlit as st
from PIL import Image
from viton_hd import run


st.markdown(
    "<h1 style='text-align: center; font-family: Chalkduster, fantasy;'>Virtual Try ON</h1>",
    unsafe_allow_html=True
)


uploaded_person = st.file_uploader("Upload a User Photo", type="jpg")
cloth = st.file_uploader("Upload a cloth Photo", type="jpg")


if uploaded_person is not None and cloth:
    person = Image.open(uploaded_person)
    # Create two columns for layout
    col1, col2 = st.columns(2)

    # Display the first image in the first column
    col1.image(person, caption="Person", width=100, use_column_width=True)

    # Display the second image in the second column
    col2.image(cloth, caption="Cloth", width=100, use_column_width=True)


    progress_bar = st.progress(0)
    # for percent_complete in range(100):
    #     time.sleep(0.40)
    #     progress_bar.progress(percent_complete + 1)
    with open('datasets/test_pairs.txt', 'w') as fopen:
        fopen.write(uploaded_person.name + ' ' + cloth.name)
    run()
    result = Image.open("result.png" )
    st.image(result , caption="Result" , width=100 , use_column_width=True)
    # st.success('Task completed!')
    st.markdown("<h3 style='text-align: center; color: green;'>Dress fitted!</h3>", unsafe_allow_html=True)







