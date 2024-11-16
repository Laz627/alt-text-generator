import streamlit as st
import pandas as pd
import requests
import base64
import re
import os
from io import BytesIO
from PIL import Image
from openai import OpenAI
import json
import time

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
    - Ensure the file extension matches the image type (e.g., `.jpg` for JPEG files).

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
    client = OpenAI(api_key=api_key)
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
skipped_files = []

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
                skipped_files.append(uploaded_file.name)
        else:
            st.error(f"Unsupported file format: {uploaded_file.name}")
            skipped_files.append(uploaded_file.name)

# Process Image URLs
if image_urls_input:
    url_list = image_urls_input.strip().split('\n')
    if len(url_list) > 20:
        st.warning("You can process a maximum of 20 images.")
        url_list = url_list[:20]
    for url in url_list:
        try:
            response = requests.get(url)
            if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                image = Image.open(BytesIO(response.content))
                images.append(image)
                original_filenames.append(url.split('/')[-1])
                image_sources.append(url)
            else:
                st.error(f"URL does not point to an image or is unreachable: {url}")
                skipped_files.append(url)
        except Exception as e:
            st.error(f"Error processing image from URL {url}: {e}")
            skipped_files.append(url)

# Limit to 20 images
if len(images) > 20:
    images = images[:20]
    original_filenames = original_filenames[:20]
    image_sources = image_sources[:20]

# Proceed if we have images, API key, and target keyword
if images and api_key and target_keyword:
    st.header("Optimizing Images...")
    for idx, image in enumerate(images):
        # Prepare image for OpenAI API
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_data = buffered.getvalue()
        base64_image = base64.b64encode(img_data).decode('utf-8')

        # Prepare the prompt with enhanced descriptions
        prompt = f"""
You are an assistant that specializes in SEO optimization for images.

Analyze the following image and generate an optimized file name and alt text for SEO purposes.

The target keyword is '{target_keyword}'.

**SEO Best Practices for Image File Names and Alt Text:**

- **File Name Guidelines:**
  - Create a unique, descriptive, and concise file name that reflects the specific content of the image.
  - Use hyphens (`-`) between words.
  - Include the target keyword naturally without keyword stuffing.
  - **Include detailed descriptive elements** such as colors, materials, objects, settings, actions, and any other relevant specifics observed in the image.
  - Keep the file name concise, ideally under 100 characters, while still being descriptive.
  - Ensure the file name has the correct file extension (e.g., `.jpg`, `.png`).

- **Alt Text Guidelines:**
  - Write a natural, **detailed**, and informative description of the image.
  - Include the target keyword naturally without keyword stuffing.
  - **Mention all unique aspects of the image** to provide full context.
  - Keep it concise, typically under 125 characters, but prioritize completeness over brevity if necessary.
  - Ensure alt text aligns with the page content for better accessibility and SEO.

**Examples:**

If the image shows a red wooden chair next to a small round table in a garden:

- Optimized Filename: `red-wooden-chair-round-table-garden-{target_keyword}.jpg`
- Alt Text: `A red wooden chair beside a small round table set in a lush garden, showcasing {target_keyword}.`

If the image shows a modern kitchen with stainless steel appliances and marble countertops:

- Optimized Filename: `modern-kitchen-stainless-steel-appliances-marble-countertops-{target_keyword}.jpg`
- Alt Text: `A modern kitchen featuring stainless steel appliances and marble countertops, highlighting {target_keyword}.`

**Provide the output exactly in the following JSON format, without any code block markers or additional text:**

{{
  "optimized_filename": "your-optimized-file-name.jpg",
  "alt_text": "Your optimized alt text here."
}}

**Note:** Only provide the JSON object. Do not include any markdown formatting, explanations, or code block markers.

Now, here's the image:
"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                ],
            }
        ]

        # Call OpenAI API
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=500,
                temperature=0,  # Ensures consistent output
            )
            # Extract response
            output = response.choices[0].message.content.strip()

            # Remove code block markers if present
            if output.startswith("```"):
                output = output.strip("```json").strip("```").strip()

            # Parse the JSON output
            try:
                result = json.loads(output)
                optimized_filename = result.get("optimized_filename", f"optimized_image_{idx+1}.png")
                alt_text = result.get("alt_text", f"An image related to {target_keyword}.")
            except json.JSONDecodeError:
                st.warning(f"Could not parse JSON response for image {idx+1}, using default values.")
                optimized_filename = f"optimized_image_{idx+1}.png"
                alt_text = f"An image related to {target_keyword}."

            # Sanitize the filename
            optimized_filename = re.sub(r'[^a-zA-Z0-9\-\.]', '', optimized_filename.replace(' ', '-').lower())

            # Truncate filename if it exceeds the maximum length
            max_filename_length = 100
            if len(optimized_filename) > max_filename_length:
                base_name, extension = os.path.splitext(optimized_filename)
                optimized_filename = f"{base_name[:max_filename_length - len(extension)]}{extension}"

            # Ensure filename uniqueness
            if optimized_filename in optimized_filenames:
                base_name, extension = os.path.splitext(optimized_filename)
                optimized_filename = f"{base_name}-{idx+1}{extension}"

            optimized_filenames.append(optimized_filename)
            alt_texts.append(alt_text)

        except Exception as e:
            st.error(f"Error optimizing image {original_filenames[idx]}: {e}")
            optimized_filenames.append(f"optimized_image_{idx+1}.png")
            alt_texts.append(f"An image related to {target_keyword}.")

        # Optional: Add a short delay to avoid rate limits
        time.sleep(1)

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

    # Display images with optimized alt text and file names in accordions
    st.header("View Optimized Images")
    for idx, image in enumerate(images):
        with st.expander(f"Image {idx + 1}: {optimized_filenames[idx]}"):
            st.image(image, caption=f"Optimized Filename: {optimized_filenames[idx]}", use_column_width=False, width=400)
            st.write(f"**Alt Text:** {alt_texts[idx]}")

    # Display any skipped files
    if skipped_files:
        st.header("Skipped Files")
        for file in skipped_files:
            st.write(f"- {file}")

else:
    if not images:
        st.info("Please upload images or provide image URLs.")
    if not target_keyword:
        st.info("Please specify a target keyword.")
