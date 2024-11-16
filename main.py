import streamlit as st
import pandas as pd
import requests
import base64
from io import BytesIO
from PIL import Image
import openai

# Title and Blurb
st.title("Image SEO Optimizer")
st.write("**Author:** Brandon Lazovic")
st.write("""
Welcome to the **Image SEO Optimizer** tool! This app helps you optimize your images for SEO by generating descriptive file names and alt text based on Google's best practices. Upload your images or provide image URLs, specify your target keyword, and let the app do the rest!
""")

# Collapsible SEO Best Practices
with st.expander("SEO Best Practices for Image File Names and Alt Text"):
    st.write("""
    **Use Descriptive and Relevant Alt Text:**
    - Write concise, informative descriptions of the image.
    - Include relevant keywords naturally without keyword stuffing.
    - Ensure alt text aligns with the page content for better accessibility and SEO.

    **Optimize Image File Names:**
    - Use short, descriptive, and keyword-rich filenames (e.g., `golden-retriever-puppy.jpg` instead of `IMG00023.JPG`).
    - Avoid generic or meaningless names like `image1.jpg` or `pic.gif`.

    **Use Supported File Formats:**
    - Preferred formats include JPEG, PNG, WebP, SVG, GIF, BMP, and AVIF.
    - Ensure the file extension matches the image type (e.g., `.jpg` for JPEG files`).

    **And More...**
    - Place images near relevant text and captions.
    - Use structured data to make images eligible for rich results in search.
    - Optimize for accessibility by including alt attributes.
    - Use responsive images with `srcset` or `<picture>` elements.
    - Compress images to balance quality and load speed.
    """)

# User Input for OpenAI API Key
st.header("Enter Your OpenAI API Key")
api_key = st.text_input("API Key", type="password")

if api_key:
    openai.api_key = api_key
else:
    st.warning("Please enter your OpenAI API key to proceed.")

# Target Keyword Input
st.header("Specify Your Target Keyword")
target_keyword = st.text_input("Target Keyword")

# Image Upload or URL Input
st.header("Upload Images or Provide Image URLs")

uploaded_files = st.file_uploader("Upload up to 20 images", type=['png', 'jpg', 'jpeg', 'webp', 'gif'], accept_multiple_files=True)
image_urls_input = st.text_area("Or enter image URLs (one per line)")

# Initialize Lists to Store Data
images = []
original_filenames = []
optimized_filenames = []
alt_texts = []
image_sources = []

# Process Uploaded Images
if uploaded_files:
    if len(uploaded_files) > 20:
        st.warning("You can upload a maximum of 20 images.")
        uploaded_files = uploaded_files[:20]
    for uploaded_file in uploaded_files:
        if uploaded_file.type in ['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/gif']:
            try:
                image = Image.open(uploaded_file)
                images.append(image)
                original_filenames.append(uploaded_file.name)
                image_sources.append(None)  # No URL for uploaded files
            except Exception as e:
                st.error(f"Error processing file {uploaded_file.name}: {e}")
        else:
            st.error(f"Unsupported file format: {uploaded_file.name}")

# Process Image URLs
if image_urls_input:
    url_list = image_urls_input.strip().split('\n')
    if len(url_list) > 20:
        st.warning("You can process a maximum of 20 images.")
        url_list = url_list[:20]
    for url in url_list:
        try:
            response = requests.get(url)
            if response.status_code == 200 and 'image' in response.headers['Content-Type']:
                image = Image.open(BytesIO(response.content))
                images.append(image)
                original_filenames.append(url.split('/')[-1])
                image_sources.append(url)
            else:
                st.error(f"URL does not point to an image or is unreachable: {url}")
        except Exception as e:
            st.error(f"Error processing image from URL {url}: {e}")

# Limit to 20 images
if len(images) > 20:
    images = images[:20]
    original_filenames = original_filenames[:20]
    image_sources = image_sources[:20]

# Proceed if we have images, API key, and target keyword
if images and api_key and target_keyword:
    st.header("Optimizing Images...")
    for idx, image in enumerate(images):
        st.image(image, caption=f"Original Filename: {original_filenames[idx]}", use_column_width=True)
        # Prepare image for OpenAI API
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        # Prepare the prompt for OpenAI
        prompt = f"""
Analyze the following image and generate an optimized file name and alt text for SEO purposes.
The target keyword is '{target_keyword}'.
- Ensure the file name is descriptive, concise, and uses hyphens between words.
- Ensure the alt text is a natural, informative description of the image, including the target keyword without keyword stuffing.

Provide the output in the following format:

File Name: [optimized_file_name]
Alt Text: [optimized_alt_text]
"""

        # Call OpenAI API
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an assistant that specializes in SEO optimization for images."},
                    {"role": "user", "content": prompt}
                ],
                files=[
                    {"name": "image.png", "bytes": img_bytes, "mime_type": "image/png"}
                ]
            )
            # Extract response
            output = response.choices[0].message['content']
            # Parse the output
            lines = output.strip().split('\n')
            file_name_line = next((line for line in lines if line.lower().startswith('file name:')), None)
            alt_text_line = next((line for line in lines if line.lower().startswith('alt text:')), None)
            if file_name_line and alt_text_line:
                optimized_filename = file_name_line.split(':', 1)[1].strip()
                alt_text = alt_text_line.split(':', 1)[1].strip()
            else:
                optimized_filename = f"optimized_image_{idx+1}.png"
                alt_text = f"An image related to {target_keyword}."
                st.warning(f"Could not parse the response for image {idx+1}, using default values.")
            optimized_filenames.append(optimized_filename)
            alt_texts.append(alt_text)

            st.write(f"**Optimized File Name:** {optimized_filename}")
            st.write(f"**Alt Text:** {alt_text}")

        except Exception as e:
            st.error(f"Error optimizing image {original_filenames[idx]}: {e}")
            optimized_filenames.append(f"optimized_image_{idx+1}.png")
            alt_texts.append(f"An image related to {target_keyword}.")

    # Provide Downloadable CSV
    st.header("Download Results")
    data = {
        "Original Filename": original_filenames,
        "Optimized Filename": optimized_filenames,
        "Alt Text": alt_texts,
        "Image URL": image_sources
    }
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='optimized_images.csv',
        mime='text/csv'
    )

else:
    if not images:
        st.info("Please upload images or provide image URLs.")
    if not target_keyword:
        st.info("Please specify a target keyword.")
