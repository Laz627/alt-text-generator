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
import zipfile
import traceback # For detailed error logging in console

# --- Configuration & Page Setup ---
st.set_page_config(layout="wide", page_title="Image SEO Optimizer")

# --- Helper Functions ---
def sanitize_filename(filename):
    """Removes invalid characters and replaces spaces with hyphens."""
    sanitized = re.sub(r'[^a-zA-Z0-9\-\.]', '', filename.replace(' ', '-').lower())
    sanitized = sanitized.strip('-.')
    sanitized = re.sub(r'-+', '-', sanitized)
    return sanitized

def truncate_filename(filename, max_length=100):
    """Truncates filename if it exceeds max_length, preserving the extension."""
    if len(filename) <= max_length:
        return filename
    base_name, extension = os.path.splitext(filename)
    available_length = max_length - len(extension)
    truncated_base = base_name[:available_length]
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
            response = requests.get(source, timeout=15)
            response.raise_for_status()
            if 'image' not in response.headers.get('Content-Type', '').lower():
                raise ValueError("URL does not point to a valid image type.")
            image = Image.open(BytesIO(response.content))
            original_filename = source.split('/')[-1].split('?')[0]
        else:
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

with st.expander("💡 ricorda: SEO Best Practices for Images"):
    st.markdown("""
    - **Alt Text:** Concise, descriptive, keyword-rich (naturally), contextually relevant. Aim for under 125 characters. Crucial for accessibility and search engines.
    - **File Names:** Short, descriptive, hyphen-separated words, keyword-rich (naturally). Avoid generic names (`IMG001.jpg`). Use `.jpg`, `.png`, `.webp`, `.gif`, `.svg`, etc.
    - **Compression:** Balance quality and file size for faster loading. JPEGs are great for photos, PNGs for graphics with transparency.
    - **Context:** Place images near relevant text. Use captions if helpful.
    - **Responsiveness:** Use `srcset` or `<picture>` for different screen sizes.
    - **Structured Data:** Use schema markup (e.g., `ImageObject`) for eligibility in rich results.
    """)

st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("1. OpenAI API Key", type="password", help="Required for generating filenames and alt text.")
target_keyword = st.sidebar.text_input("2. Target Keyword", help="Primary keyword to guide optimization.")
project_number = st.sidebar.text_input("3. Project Number (Optional)", help="Appended to filenames (e.g., 'image-slug-PN123.jpg'). Alphanumeric and hyphens allowed.")
compression_quality = st.sidebar.slider("4. Compression Quality (JPEG Output)", 1, 100, 85, help="Adjust the quality (and file size) of the output JPEG images. Higher quality means larger files.")

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
    uploaded_files = st.file_uploader("Upload Images", type=['png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp', 'avif'], accept_multiple_files=True, help="You can upload up to 20 images.")
with col2:
    image_urls_input = st.text_area("Or Enter Image URLs (one per line)", height=150, help="Enter direct URLs to image files.")

image_sources_input = []
if uploaded_files: image_sources_input.extend(uploaded_files)
if image_urls_input:
    urls = [url.strip() for url in image_urls_input.strip().split('\n') if url.strip()]
    image_sources_input.extend(urls)

total_images = len(image_sources_input)
processed_data = []
skipped_files = []
processing_errors = []

if not api_key: st.warning("🚨 Please enter your OpenAI API key in the sidebar to begin.")
elif not target_keyword: st.warning("🎯 Please enter a target keyword in the sidebar.")
elif not image_sources_input: st.info("➕ Please upload images or provide URLs.")
elif total_images > 20: st.error(f"❌ Too many images ({total_images}). Please provide a maximum of 20 images.")
else:
    if st.button("✨ Optimize Images", type="primary"):
        if not target_keyword:
             st.error("❌ Target keyword cannot be empty when optimizing.")
             st.stop()

        client = OpenAI(api_key=api_key)
        st.header("⏳ Processing Images...")
        progress_bar = st.progress(0, text="Initializing...")
        start_time = time.time()
        optimized_filenames_set = set()

        for idx, source in enumerate(image_sources_input):
            progress_text = f"Processing image {idx + 1} of {total_images}..."
            progress_bar.progress((idx + 1) / total_images, text=progress_text)
            image, original_filename, error = get_image_from_source(source)
            source_identifier = source if isinstance(source, str) else source.name

            if error:
                skipped_files.append(f"{source_identifier} (Loading Error)")
                processing_errors.append(error); continue
            if not image or not original_filename:
                 skipped_files.append(f"{source_identifier} (Load Failed)")
                 processing_errors.append(f"Failed to load image data for {source_identifier}"); continue

            try:
                compressed_buffer = BytesIO()
                img_for_compression = image.copy()
                if img_for_compression.mode in ('RGBA', 'LA', 'P'):
                     img_for_compression = img_for_compression.convert('RGB')
                img_for_compression.save(compressed_buffer, format="JPEG", quality=compression_quality, optimize=True)
                compressed_image_data = compressed_buffer.getvalue()

                openai_image_buffer = BytesIO()
                image.save(openai_image_buffer, format="PNG")
                base64_image = base64.b64encode(openai_image_buffer.getvalue()).decode('utf-8')

                sanitized_keyword_for_api = target_keyword
                try:
                    if isinstance(target_keyword, str):
                        ascii_encoded_keyword = target_keyword.encode('ascii', 'ignore')
                        sanitized_keyword_for_api = ascii_encoded_keyword.decode('ascii')
                        if sanitized_keyword_for_api != target_keyword:
                            print(f"Console Log: Target keyword '{target_keyword}' sanitized to '{sanitized_keyword_for_api}' for API for image {idx+1} ({original_filename}).")
                    else:
                         print(f"Console Log: Warning - target_keyword was not string for image {idx+1}. Using as is.")
                         sanitized_keyword_for_api = str(target_keyword)
                except Exception as e:
                    print(f"Console Log: Error during keyword sanitization: {e}. Using original: '{target_keyword}'")

                prompt_text_for_api = f"""
Analyze the image and generate an SEO-optimized file name (without extension initially) and alt text.
Target Keyword: '{sanitized_keyword_for_api}'

Guidelines:
- Filename: Descriptive, concise, hyphen-separated, naturally include keyword and key visual elements (objects, colors, setting, actions). NO file extension. Be specific but avoid excessive length.
- Alt Text: Natural, descriptive, max 125 chars ideally, include keyword, describe key visual elements providing context. Be specific.

Output ONLY in this exact JSON format:
{{
  "base_filename": "your-optimized-base-file-name",
  "alt_text": "Your optimized alt text."
}}
"""
                final_sanitized_prompt_text = prompt_text_for_api
                try:
                    encoded_prompt = prompt_text_for_api.encode('ascii', 'ignore')
                    final_sanitized_prompt_text = encoded_prompt.decode('ascii')
                    if final_sanitized_prompt_text != prompt_text_for_api:
                        print(f"Console Log: Full prompt text was further sanitized for API for image {idx+1} ({original_filename}).")
                except Exception as e_prompt_sanitize:
                    print(f"Console Log: Error during full prompt sanitization: {e_prompt_sanitize}. Using as constructed.")
                
                print(f"\n--- DEBUG: Content being sent for image {idx+1} ({original_filename}) ---")
                print(f"Target Keyword originally: '{target_keyword}'")
                print(f"Sanitized Keyword for API in prompt construction: '{sanitized_keyword_for_api}'")
                # print(f"Final text content for API message (after full sanitization):\n{final_sanitized_prompt_text}") # Keep this for full debug if needed
                print(f"First 100 chars of final text for API: {final_sanitized_prompt_text[:100]}...") # Print snippet
                print("--- END DEBUG Info ---\n")

                messages = [
                    {"role": "user", "content": [
                        {"type": "text", "text": final_sanitized_prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]}
                ]

                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini", messages=messages, max_tokens=150, temperature=0.1,
                        response_format={"type": "json_object"}
                    )
                    output = response.choices[0].message.content.strip()
                    result = json.loads(output)
                    base_filename = result.get("base_filename", f"optimized-image-{idx+1}")
                    alt_text = result.get("alt_text", f"Image related to {target_keyword}")
                except Exception as api_e:
                    error_detail = str(api_e)
                    if isinstance(api_e, UnicodeEncodeError):
                        error_detail += (f" This often happens with non-ASCII characters (like ® ™ ©) in inputs. "
                                         f"Original Target Keyword: '{target_keyword}'. "
                                         f"Keyword used in prompt construction (sanitized): '{sanitized_keyword_for_api}'. "
                                         f"The entire text content sent to API was also sanitized (see console for 'Final text content...'). "
                                         f"If all sanitized versions in the log appear correct (no ®), the issue is likely deep within the request library's encoding of the entire request payload or headers, or an environment setting.")
                    elif "response_format" in error_detail.lower():
                        error_detail += " API trouble generating JSON. Check prompt/model limits."
                    st.error(f"⚠️ OpenAI API error for image {idx+1} ({original_filename}): {error_detail}. Using defaults.")
                    processing_errors.append(f"OpenAI API error for {original_filename}: {error_detail}")
                    base_filename = f"api-error-{idx+1}-{sanitized_project_number if sanitized_project_number else ''}".rstrip('-')
                    alt_text = f"Image related to {target_keyword} (API error)"

                final_base_filename = sanitize_filename(base_filename)
                if sanitized_project_number: final_base_filename += f"-{sanitized_project_number}"
                final_filename_with_ext = f"{final_base_filename}.jpg"
                final_filename_with_ext = truncate_filename(final_filename_with_ext)

                unique_filename = final_filename_with_ext; counter = 1
                while unique_filename in optimized_filenames_set:
                    name, ext = os.path.splitext(final_filename_with_ext)
                    name_part_to_strip = f"-{counter-1}"
                    if name.endswith(name_part_to_strip): name = name[:-len(name_part_to_strip)]
                    if not name: name = f"duplicate-base-{idx+1}"
                    unique_filename = f"{name}-{counter}{ext}"; counter += 1
                optimized_filenames_set.add(unique_filename)

                processed_data.append({
                    "original_filename": original_filename, "optimized_filename": unique_filename,
                    "alt_text": alt_text, "image_url": source if isinstance(source, str) else None,
                    "compressed_data": compressed_image_data, "pil_image": image
                })
            except Exception as process_e:
                 skipped_files.append(f"{source_identifier} (Processing Error)")
                 processing_errors.append(f"Error processing {original_filename}: {process_e}")
                 print(f"--- Error processing {original_filename} ---"); traceback.print_exc(); print("--- End of Error ---")
                 continue
            time.sleep(0.5)

        progress_bar.empty(); end_time = time.time()
        st.success(f"✅ Optimization complete for {len(processed_data)} of {total_images} images in {end_time - start_time:.2f} seconds!")

        if processed_data:
            st.header("📊 Results & Downloads")
            display_df_data = [{"Original Filename":i["original_filename"],"Optimized Filename":i["optimized_filename"],"Alt Text":i["alt_text"],"Original Source":i["image_url"] if i["image_url"] else "Uploaded"} for i in processed_data]
            if display_df_data: st.dataframe(pd.DataFrame(display_df_data))
            else: st.info("No data to display in table.")

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                if display_df_data:
                    csv = pd.DataFrame(display_df_data).to_csv(index=False).encode('utf-8')
                    st.download_button("📥 CSV Summary", csv, 'image_seo_summary.csv', 'text/csv', key='csv_download')
            with col_dl2:
                if processed_data:
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for item in processed_data: zf.writestr(item["optimized_filename"], item["compressed_data"])
                    st.download_button("📦 Optimized Images (.zip)", zip_buffer.getvalue(), 'optimized_images.zip', 'application/zip', key='zip_download')

            st.header("🖼️ Optimized Image Preview")
            if processed_data:
                for i_disp, item in enumerate(processed_data):
                    with st.expander(f"Image {i_disp + 1}: {item.get('optimized_filename', 'N/A')}", expanded=False):
                        if 'pil_image' in item: st.image(item["pil_image"], caption=f"Optimized Filename: {item.get('optimized_filename', 'N/A')}", width=300)
                        else: st.warning("Preview not available.")
                        st.markdown(f"**Alt Text:**")
                        st.text_area("Generated Alt Text", value=item.get('alt_text', 'N/A'), height=75, key=f"alt_{i_disp}", disabled=True)
            else: st.info("No images processed to preview.")

        if skipped_files:
            st.header("⚠️ Skipped / Errored Items")
            for i, file_info in enumerate(skipped_files):
                st.warning(f"- {file_info}")
                if i < len(processing_errors): st.error(f"  Error details: {processing_errors[i]}", icon="🐛")

st.markdown("---"); st.markdown("Built with ❤️ using Streamlit & OpenAI")
