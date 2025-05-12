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
import zipfile # Added for zip functionality

# --- Configuration & Page Setup ---
st.set_page_config(layout="wide", page_title="Image SEO Optimizer")

# --- Helper Functions ---
def sanitize_filename(filename):
    """Removes invalid characters and replaces spaces with hyphens."""
    # Remove invalid characters (allow letters, numbers, hyphens, periods)
    sanitized = re.sub(r'[^a-zA-Z0-9\-\.]', '', filename.replace(' ', '-').lower())
    # Remove leading/trailing hyphens and periods
    sanitized = sanitized.strip('-.')
    # Replace multiple hyphens with a single hyphen
    sanitized = re.sub(r'-+', '-', sanitized)
    return sanitized

def truncate_filename(filename, max_length=100):
    """Truncates filename if it exceeds max_length, preserving the extension."""
    if len(filename) <= max_length:
        return filename
    base_name, extension = os.path.splitext(filename)
    # Calculate how much of the base name we can keep
    available_length = max_length - len(extension)
    truncated_base = base_name[:available_length]
    # Ensure it doesn't end with a hyphen after truncation
    truncated_base = truncated_base.rstrip('-')
    return f"{truncated_base}{extension}"

def get_image_from_source(source):
    """Loads a PIL Image from either an uploaded file or a URL."""
    image = None
    error = None
    original_filename = None
    is_url = isinstance(source, str)

    try:
        if is_url:
            # Process URL
            response = requests.get(source, timeout=15) # Added timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            if 'image' not in response.headers.get('Content-Type', '').lower():
                raise ValueError("URL does not point to a valid image type.")
            image = Image.open(BytesIO(response.content))
            original_filename = source.split('/')[-1].split('?')[0] # Basic filename extraction
        else:
            # Process Uploaded File
            if source.type not in ['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/gif', 'image/bmp', 'image/avif']:
                 raise ValueError(f"Unsupported file format: {source.type}")
            image = Image.open(source)
            original_filename = source.name
    except requests.exceptions.RequestException as e:
        error = f"Error fetching URL {source}: {e}"
    except ValueError as e:
        error = f"Error processing {'URL' if is_url else 'file'} {source if is_url else source.name}: {e}"
    except Exception as e:
        error = f"Unexpected error processing {'URL' if is_url else 'file'} {source if is_url else source.name}: {e}"

    return image, original_filename, error


# --- Main App UI ---
st.title("🚀 Image SEO Optimizer")
st.write("**Author:** Brandon Lazovic")
st.markdown("""
Welcome to the **Image SEO Optimizer**! This tool enhances your images for SEO by:
- Generating descriptive file names and alt text using AI (powered by OpenAI).
- Appending a project number to filenames.
- Compressing images (output as JPEG) with adjustable quality.
- Providing optimized assets in a downloadable zip file.

Upload images or provide URLs, set your parameters, and optimize!
""")

# Collapsible SEO Best Practices
with st.expander("💡 ricorda: SEO Best Practices for Images"):
    st.markdown("""
    - **Alt Text:** Concise, descriptive, keyword-rich (naturally), contextually relevant. Aim for under 125 characters. Crucial for accessibility and search engines.
    - **File Names:** Short, descriptive, hyphen-separated words, keyword-rich (naturally). Avoid generic names (`IMG001.jpg`). Use `.jpg`, `.png`, `.webp`, `.gif`, `.svg`, etc.
    - **Compression:** Balance quality and file size for faster loading. Use tools or save-for-web options. JPEGs are great for photos, PNGs for graphics with transparency.
    - **Context:** Place images near relevant text. Use captions if helpful.
    - **Responsiveness:** Use `srcset` or `<picture>` for different screen sizes.
    - **Structured Data:** Use schema markup (e.g., `ImageObject`) for eligibility in rich results.
    """)

st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("1. OpenAI API Key", type="password", help="Required for generating filenames and alt text.")
target_keyword = st.sidebar.text_input("2. Target Keyword", help="Primary keyword to guide optimization.")
project_number = st.sidebar.text_input("3. Project Number (Optional)", help="Appended to filenames (e.g., 'image-slug-PN123.jpg'). Alphanumeric and hyphens allowed.")
compression_quality = st.sidebar.slider("4. Compression Quality (JPEG Output)", 1, 100, 85, help="Adjust the quality (and file size) of the output JPEG images. Higher quality means larger files.")

# Sanitize project number input
sanitized_project_number = ""
if project_number:
    sanitized_project_number = re.sub(r'[^a-zA-Z0-9\-]', '', project_number).strip('-')
    if project_number != sanitized_project_number:
       st.sidebar.warning(f"Project number sanitized to: `{sanitized_project_number}`")
    if not sanitized_project_number:
        st.sidebar.warning("Invalid project number entered after sanitization, it will not be used.")


st.header("🖼️ Image Input (Max 20)")
col1, col2 = st.columns(2)

with col1:
    uploaded_files = st.file_uploader(
        "Upload Images",
        type=['png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp', 'avif'], # Added more types
        accept_multiple_files=True,
        help="You can upload up to 20 images."
    )

with col2:
    image_urls_input = st.text_area(
        "Or Enter Image URLs (one per line)",
        height=150,
        help="Enter direct URLs to image files."
    )

# --- Initialization ---
image_sources_input = []
if uploaded_files:
    image_sources_input.extend(uploaded_files)
if image_urls_input:
    urls = [url.strip() for url in image_urls_input.strip().split('\n') if url.strip()]
    image_sources_input.extend(urls)

total_images = len(image_sources_input)
processed_data = [] # List to store dicts for each processed image
skipped_files = []
processing_errors = []

# --- Validation and Processing Trigger ---
if not api_key:
    st.warning("🚨 Please enter your OpenAI API key in the sidebar to begin.")
elif not target_keyword:
    st.warning("🎯 Please enter a target keyword in the sidebar.")
elif not image_sources_input:
    st.info("➕ Please upload images or provide URLs.")
elif total_images > 20:
    st.error(f"❌ Too many images ({total_images}). Please provide a maximum of 20 images.")
else:
    # Process Button
    if st.button("✨ Optimize Images", type="primary"):
        client = OpenAI(api_key=api_key)
        st.header("⏳ Processing Images...")
        progress_bar = st.progress(0, text="Initializing...")
        start_time = time.time()

        optimized_filenames_set = set() # For ensuring uniqueness

        for idx, source in enumerate(image_sources_input):
            progress_text = f"Processing image {idx + 1} of {total_images}..."
            progress_bar.progress((idx + 1) / total_images, text=progress_text)

            image, original_filename, error = get_image_from_source(source)
            source_identifier = source if isinstance(source, str) else source.name

            if error:
                skipped_files.append(f"{source_identifier} (Loading Error)")
                processing_errors.append(error)
                continue # Skip to the next image

            if not image or not original_filename:
                 skipped_files.append(f"{source_identifier} (Load Failed)")
                 processing_errors.append(f"Failed to load image data for {source_identifier}")
                 continue

            try:
                # 1. Compress Image (Outputting as JPEG)
                compressed_buffer = BytesIO()
                # Convert to RGB if it has transparency (alpha channel) for JPEG saving
                if image.mode in ('RGBA', 'LA', 'P'):
                     image = image.convert('RGB')
                image.save(compressed_buffer, format="JPEG", quality=compression_quality, optimize=True)
                compressed_image_data = compressed_buffer.getvalue()

                # 2. Prepare image for OpenAI (Use original or a PNG version for better analysis)
                openai_image_buffer = BytesIO()
                # Convert to RGB before saving as PNG if it's palette-based ('P') with transparency
                # This avoids potential issues with OpenAI interpreting some PNG palettes
                temp_image_for_openai = image
                if temp_image_for_openai.mode == 'P':
                     temp_image_for_openai = temp_image_for_openai.convert('RGBA') # Use RGBA to preserve transparency info if any
                elif temp_image_for_openai.mode == 'LA':
                     temp_image_for_openai = temp_image_for_openai.convert('RGBA') # Convert Luminance+Alpha too

                # If image wasn't originally RGB (like RGBA, LA), convert to RGB *before* saving as PNG for OpenAI.
                # This seems counter-intuitive, but helps avoid issues with how the Vision model interprets alpha channels sometimes.
                # We already converted the main 'image' variable to RGB for JPEG saving if needed.
                # Let's re-evaluate: Send PNG *with* transparency if present. Revert the conversion here.
                # image.save(openai_image_buffer, format="PNG") # Send PNG to OpenAI - keep original mode
                
                # Let's stick to sending PNG, trying to preserve original mode where possible.
                # Pillow handles PNG saving correctly for various modes.
                image.save(openai_image_buffer, format="PNG")
                base64_image = base64.b64encode(openai_image_buffer.getvalue()).decode('utf-8')


                # 3. Call OpenAI API for filename and alt text
                prompt = f"""
Analyze the image and generate an SEO-optimized file name (without extension initially) and alt text.
Target Keyword: '{target_keyword}'

Guidelines:
- Filename: Descriptive, concise, hyphen-separated, naturally include keyword and key visual elements (objects, colors, setting, actions). NO file extension. Be specific but avoid excessive length.
- Alt Text: Natural, descriptive, max 125 chars ideally, include keyword, describe key visual elements providing context. Be specific.

Output ONLY in this exact JSON format:
{{
  "base_filename": "your-optimized-base-file-name",
  "alt_text": "Your optimized alt text."
}}
"""
                messages = [
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]}
                ]

                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini", # Using the cheaper/faster mini model
                        messages=messages,
                        max_tokens=150, # Reduced tokens needed
                        temperature=0.1, # Slightly more creative but still consistent
                        response_format={"type": "json_object"} # Enforce JSON output
                    )
                    output = response.choices[0].message.content.strip()
                    result = json.loads(output)
                    base_filename = result.get("base_filename", f"optimized-image-{idx+1}")
                    alt_text = result.get("alt_text", f"Image related to {target_keyword}")

                except Exception as api_e:
                    st.error(f"⚠️ OpenAI API error for image {idx+1}: {api_e}. Using default names.")
                    processing_errors.append(f"OpenAI API error for {original_filename}: {api_e}")
                    base_filename = f"api-error-image-{idx+1}"
                    alt_text = f"Image related to {target_keyword} (API error)"

                # 4. Construct Final Filename
                # Sanitize AI-generated base filename
                final_base_filename = sanitize_filename(base_filename)

                # Append project number if provided
                if sanitized_project_number:
                    final_base_filename += f"-{sanitized_project_number}"

                # Add extension (always .jpg because we compress to JPEG)
                final_filename_with_ext = f"{final_base_filename}.jpg"

                # Truncate if necessary
                final_filename_with_ext = truncate_filename(final_filename_with_ext)

                # Ensure uniqueness
                unique_filename = final_filename_with_ext
                counter = 1
                while unique_filename in optimized_filenames_set:
                    name, ext = os.path.splitext(final_filename_with_ext)
                    # Avoid adding number if it already ends with -number.ext
                    if name.endswith(f"-{counter-1}"):
                         name = name[:-(len(str(counter-1))+1)] # Adjust slicing index
                    # Ensure name doesn't become empty after stripping counter part
                    if not name: name = f"duplicate-{idx+1}" # Fallback if name becomes empty
                    unique_filename = f"{name}-{counter}{ext}"
                    counter += 1
                optimized_filenames_set.add(unique_filename)


                # 5. Store results
                processed_data.append({
                    "original_filename": original_filename,
                    "optimized_filename": unique_filename,
                    "alt_text": alt_text,
                    "image_url": source if isinstance(source, str) else None, # Store URL if applicable
                    "compressed_data": compressed_image_data,
                    "pil_image": image # Keep original PIL object for display
                })

            except Exception as process_e:
                 skipped_files.append(f"{source_identifier} (Processing Error)")
                 processing_errors.append(f"Error processing {original_filename}: {process_e}")
                 # Also log the traceback for debugging in the console
                 import traceback
                 print(f"Error processing {original_filename}:")
                 traceback.print_exc()
                 continue # Skip to next image

            # Optional delay
            time.sleep(0.5) # Reduced delay

        progress_bar.empty() # Remove progress bar
        end_time = time.time()
        st.success(f"✅ Optimization complete for {len(processed_data)} images in {end_time - start_time:.2f} seconds!")

        # --- Display Results and Downloads ---
        if processed_data:
            st.header("📊 Results & Downloads")

            # Prepare data for display and CSV
            display_df = pd.DataFrame([{
                "Original Filename": item["original_filename"],
                "Optimized Filename": item["optimized_filename"],
                "Alt Text": item["alt_text"],
                "Original Source": item["image_url"] if item["image_url"] else "Uploaded"
            } for item in processed_data])

            st.dataframe(display_df) # Display the dataframe

            col_dl1, col_dl2 = st.columns(2)

            # CSV Download
            with col_dl1:
                csv = display_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV Summary",
                    data=csv,
                    file_name='image_seo_summary.csv',
                    mime='text/csv',
                    key='csv_download'
                )

            # Zip File Download
            with col_dl2:
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for item in processed_data:
                        zip_file.writestr(item["optimized_filename"], item["compressed_data"])

                st.download_button(
                    label="📦 Download Optimized Images (.zip)",
                    data=zip_buffer.getvalue(),
                    file_name='optimized_images.zip',
                    mime='application/zip',
                    key='zip_download'
                )

            # Display Images
            st.header("🖼️ Optimized Image Preview")
            if processed_data: # Check if there's data before trying to display
                for idx, item in enumerate(processed_data):
                     # Use a unique key for each expander
                    with st.expander(f"Image {idx + 1}: {item.get('optimized_filename', 'N/A')}", expanded=False):
                        # Check if 'pil_image' exists before displaying
                        if 'pil_image' in item:
                            st.image(
                                item["pil_image"], # Display original PIL object for preview
                                caption=f"Optimized Filename: {item.get('optimized_filename', 'N/A')}",
                                width=300 # Set specific width
                            )
                        else:
                            st.warning("Preview not available for this item.")

                        st.markdown(f"**Alt Text:**")
                        # Use get with a default value for safety
                        st.text_area("Generated Alt Text", value=item.get('alt_text', 'N/A'), height=75, key=f"alt_{idx}", disabled=True)
            else:
                st.info("No images were successfully processed to preview.")


        # Display Skipped/Errored Files
        if skipped_files:
            st.header("⚠️ Skipped / Errored Items")
            # Use a set to show unique errors if they are duplicated
            unique_errors = set(processing_errors)
            for i, file_info in enumerate(skipped_files):
                st.warning(f"- {file_info}")
                # Display corresponding error if available (simple matching by index for now)
                if i < len(processing_errors):
                     st.error(f"  Error: {processing_errors[i]}", icon="🐛")
            # Optionally print unique errors encountered if list is long
            # if len(unique_errors) > 5:
            #    st.subheader("Unique Errors Encountered:")
            #    for error_msg in unique_errors:
            #        st.caption(error_msg)


# --- Footer ---
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & OpenAI")
