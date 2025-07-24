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
import traceback

# Ensure streamlit-image-comparison is installed
try:
    from streamlit_image_comparison import image_comparison
    image_comparison_available = True
except ModuleNotFoundError:
    image_comparison_available = False

# --- Configuration & Page Setup ---
st.set_page_config(layout="wide", page_title="Image SEO Optimizer")

# --- Helper Functions ---
def sanitize_filename(filename):
    """Sanitizes a string to be a valid filename, preserving the extension."""
    base, ext = os.path.splitext(filename)
    sanitized_base = base.lower().replace(' ', '-')
    sanitized_base = re.sub(r'[^a-z0-9\-]', '', sanitized_base)
    sanitized_base = re.sub(r'-+', '-', sanitized_base)
    sanitized_base = sanitized_base.strip('-')
    if not ext:
        ext = '.webp'
    return f"{sanitized_base}{ext}"

def truncate_filename(filename, max_length=80):
    """Truncates a filename to a max length, preserving the extension."""
    if len(filename) <= max_length: return filename
    base_name, extension = os.path.splitext(filename)
    available_length = max_length - len(extension)
    truncated_base = base_name[:available_length]
    truncated_base = truncated_base.rstrip('-')
    return f"{truncated_base}{extension}"

def get_image_from_source(source):
    """Fetches and opens an image from a URL or an uploaded file."""
    image, error, original_filename, original_data = None, None, None, None
    is_url = isinstance(source, str)
    try:
        if is_url:
            response = requests.get(source, timeout=15)
            response.raise_for_status()
            if 'image' not in response.headers.get('Content-Type', '').lower():
                raise ValueError("URL does not point to a valid image type.")
            original_data = response.content
            image = Image.open(BytesIO(original_data))
            original_filename = source.split('/')[-1].split('?')[0]
        else: # UploadedFile object
            original_data = source.getvalue()
            source.seek(0)
            if source.type not in ['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/gif', 'image/bmp', 'image/avif']:
                raise ValueError(f"Unsupported file format: {source.type}")
            image = Image.open(source)
            original_filename = source.name
    except Exception as e:
        error = f"Error processing {'URL' if is_url else 'file'}: {e}"
    return image, original_filename, error, original_data

def format_size(size_bytes):
    """Formats bytes into a human-readable string (KB or MB)."""
    if size_bytes is None or size_bytes < 0: return "N/A"
    if size_bytes == 0: return "0 B"
    size_kb = size_bytes / 1024
    if size_kb < 1024:
        return f"{size_kb:.1f} KB"
    size_mb = size_kb / 1024
    return f"{size_mb:.1f} MB"

def _compress_image(image, quality):
    """Compresses a PIL image to WebP format."""
    compressed_buffer = BytesIO()
    image.save(compressed_buffer, format="WEBP", quality=quality)
    compressed_image_data = compressed_buffer.getvalue()
    return compressed_image_data, len(compressed_image_data)

def _generate_ai_metadata(image, client, config):
    """Generates filename and alt text using OpenAI API."""
    openai_image_buffer = BytesIO()
    image.save(openai_image_buffer, format="PNG")
    base64_image = base64.b64encode(openai_image_buffer.getvalue()).decode('utf-8')
    has_filename_context = config['product_type'] or config['city_geo_target']
    has_alt_text_context = config['service_type'] or config['product_type'] or config['city_geo_target'] or config['additional_context']
    filename_instructions = """
- Use the **Product Type** as the primary basis for the filename.
- Enhance it with specific nouns from the **Primary Keyword** if they add necessary detail (e.g., 'patio-door').
- Include the **Location**. Do **NOT** include the Service Type or Additional Context.
- **GOOD Example:** `lifestyle-series-picture-windows-patio-door-salina-ks`
""" if has_filename_context else """
- **Analyze the image for its main visual subject** (e.g., 'bay window', 'tan siding').
- Build the filename **primarily from these visual details**.
- **GOOD Example:** `tan-siding-bay-window-exterior`
"""
    alt_text_instructions = """
- Tell a short, descriptive story about the image. Start with the main visual subject, then weave in the context from the background information.
- The final text **MUST be a single, concise sentence under 125 characters.**
- **GOOD Example:** "A two-story wall of Pella Lifestyle Series picture windows and a new patio door, shown after a full replacement project in Salina, KS."
""" if has_alt_text_context else """
- Identify 2-3 of the most important visual details in the image (e.g., window arrangement, door style, siding color).
- Combine these details into a descriptive, human-sounding sentence that also includes the **Primary Keyword**.
- The final text **MUST be a single, concise sentence under 125 characters.**
- **GOOD Example:** "Large picture windows above a three-panel sliding glass door on a home with tan siding, providing a modern exterior."
"""
    prompt_text_for_api = f"""
Your task is to analyze the image and generate a single, valid JSON object with `base_filename` and `alt_text`.
**BACKGROUND INFORMATION:**
- Primary Keyword: '{config['keyword']}'
- Service Type: '{config['service_type'] or "Not provided"}'
- Product Type: '{config['product_type'] or "Not provided"}'
- Location: '{config['city_geo_target'] or "Not provided"}'
- Additional Project Context: '{config['additional_context'] or "Not provided"}'
**YOUR INSTRUCTIONS:**
1.  **For `base_filename`**:
{filename_instructions}
2.  **For `alt_text`**:
{alt_text_instructions}
**IMPORTANT: Output ONLY the final, polished JSON object.**
```json
{{
  "base_filename": "your-descriptive-hyphenated-name",
  "alt_text": "Your natural, human-sounding alt text."
}}```
"""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text_for_api}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}]
    base_filename_from_api, alt_text, api_error = f"api-error-processing", f"Image related to {config['keyword']}", None
    try:
        response = client.chat.completions.create(model="gpt-4o", messages=messages, max_tokens=200, temperature=0.4, response_format={"type": "json_object"})
        output = response.choices[0].message.content.strip()
        result = json.loads(output)
        base_filename_from_api = result.get("base_filename", f"optimized-image-fallback")
        alt_text = result.get("alt_text", f"Image of {config['keyword']}")
        if base_filename_from_api.endswith(('.webp', '.png', '.jpg')):
             base_filename_from_api = os.path.splitext(base_filename_from_api)[0]
    except Exception as api_e:
        api_error = f"OpenAI API Err: {api_e}"
    return base_filename_from_api, alt_text, api_error

def _finalize_filename(base_name, proj_num_sanitized, existing_filenames):
    """Takes a base name and creates a final, unique, sanitized filename."""
    sanitized_base_name = sanitize_filename(base_name).replace('.webp', '')
    filename_core = sanitized_base_name
    if proj_num_sanitized:
        filename_core += f"-{proj_num_sanitized}"
    final_filename_with_ext = truncate_filename(f"{filename_core}.webp")
    
    unique_filename = final_filename_with_ext
    counter = 1
    while unique_filename in existing_filenames:
        core_name_no_ext, ext = os.path.splitext(final_filename_with_ext)
        if counter > 1 and core_name_no_ext.endswith(f'-{counter-1}'):
             core_name_no_ext = core_name_no_ext[:-len(f'-{counter-1}')]
        unique_filename = f"{core_name_no_ext}-{counter}{ext}"
        counter += 1
    return unique_filename

def create_title_from_filename(filename):
    """Creates a capitalized title from a hyphenated filename."""
    base_name = os.path.splitext(filename)[0]
    return base_name.replace('-', ' ').title()

def process_image(source, client, config, existing_filenames=None):
    """Processes a single image: fetches, compresses, and gets AI metadata."""
    if existing_filenames is None: existing_filenames = set()
    image, original_filename, error, original_image_data = get_image_from_source(source)
    source_identifier = source if isinstance(source, str) else source.name

    if error: return {"status": "error", "message": error, "original_filename": source_identifier}
    if not image or not original_filename or original_image_data is None: return {"status": "error", "message": "Failed to load image data.", "original_filename": source_identifier}

    try:
        original_size_bytes = len(original_image_data)
        compressed_image_data, compressed_size_bytes = _compress_image(image, config['compression_quality'])
        savings_percentage = ((original_size_bytes - compressed_size_bytes) / original_size_bytes) * 100 if original_size_bytes > 0 else 0.0
        
        base_filename_from_api, alt_text, api_error = _generate_ai_metadata(image, client, config)
        unique_filename = _finalize_filename(base_filename_from_api, config['sanitized_project_number'], existing_filenames)
        image_title = create_title_from_filename(unique_filename)
        
        return {
            "status": "success", "original_filename": original_filename, "optimized_filename": unique_filename,
            "image_title": image_title, "alt_text": alt_text, "original_size_bytes": original_size_bytes, 
            "compressed_size_bytes": compressed_size_bytes, "savings_percentage": savings_percentage, 
            "image_url": source if isinstance(source, str) else None, "compressed_data": compressed_image_data, 
            "original_data": original_image_data, "api_error": api_error
        }
    except Exception as process_e:
         return {"status": "error", "message": f"Error during processing: {process_e}", "original_filename": original_filename}

# --- Main App UI ---
st.title("Image SEO Optimizer")
st.markdown("""Optimize images for SEO: AI filenames/alt text, WebP compression, and flexible download capabilities.""")

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("1. OpenAI API Key", type="password", help="Required for AI generation.")
keyword = st.sidebar.text_input("2. Keyword", help="Primary keyword for optimization (e.g., 'home exterior').")
service_type = st.sidebar.text_input("3. Service Type (Optional)", help="e.g., 'window replacement'.")
product_type = st.sidebar.text_input("4. Product Type (Optional)", help="e.g., 'Pella casement windows'.")
city_geo_target = st.sidebar.text_input("5. City / GEO Target (Optional)", help="e.g., 'Topeka, KS'.")
project_number = st.sidebar.text_input("6. Project Number (Optional)", help="Appended to filenames (e.g., image-slug-PN123.webp).")
additional_context = st.sidebar.text_area("7. Additional Context (Optional)", help="Provide unique details like 'before photo' or 'removed old window grilles'. This will be used in the alt text.")
compression_quality = st.sidebar.slider("8. WebP Compression Quality", 1, 100, 80, help="Adjust WebP quality. Higher = better quality, larger file.")
sanitized_project_number = ""
if project_number:
    sanitized_project_number = re.sub(r'[^a-zA-Z0-9\-]', '', project_number).strip('-')
config = {"keyword": keyword, "service_type": service_type, "product_type": product_type, "city_geo_target": city_geo_target, "project_number": project_number, "additional_context": additional_context, "compression_quality": compression_quality, "sanitized_project_number": sanitized_project_number}

# --- Initialize Session State ---
if 'bulk_processed_data' not in st.session_state: st.session_state.bulk_processed_data = []
if 'bulk_compare_index' not in st.session_state: st.session_state.bulk_compare_index = None
if 'single_processed_item' not in st.session_state: st.session_state.single_processed_item = None
if 'single_compare_active' not in st.session_state: st.session_state.single_compare_active = False

# --- Callback Functions for Editing ---
def update_bulk_filename(index):
    new_filename = st.session_state[f"bulk_fn_{index}"]
    st.session_state.bulk_processed_data[index]['optimized_filename'] = sanitize_filename(new_filename)

def update_bulk_alt_text(index):
    st.session_state.bulk_processed_data[index]['alt_text'] = st.session_state[f"bulk_alt_{index}"]

def update_bulk_title(index):
    st.session_state.bulk_processed_data[index]['image_title'] = st.session_state[f"bulk_title_{index}"]

def update_single_filename():
    new_filename = st.session_state["single_fn_edit"]
    st.session_state.single_processed_item['optimized_filename'] = sanitize_filename(new_filename)

def update_single_alt_text():
    st.session_state.single_processed_item['alt_text'] = st.session_state["single_alt_edit"]

def update_single_title():
    st.session_state.single_processed_item['image_title'] = st.session_state["single_title_edit"]


# --- Tabbed Interface ---
tab1, tab2 = st.tabs(["Bulk Image Optimizer", "Singular Image Optimizer"])

# --- BULK OPTIMIZER TAB ---
with tab1:
    st.header("🖼️ Bulk Image Input (Max 20)")
    col1_in, col2_in = st.columns(2)
    with col1_in:
        uploaded_files = st.file_uploader("Upload Images", type=['png','jpg','jpeg','webp','gif','bmp','avif'], accept_multiple_files=True, help="Max 20 images.", key="bulk_uploader")
    with col2_in:
        image_urls_input = st.text_area("Or Enter Image URLs (one per line)", height=150, help="Direct image URLs.", key="bulk_urls")
    
    image_sources_input = []
    if uploaded_files: image_sources_input.extend(uploaded_files)
    if image_urls_input: image_sources_input.extend([url.strip() for url in image_urls_input.strip().split('\n') if url.strip()])
    total_images = len(image_sources_input)

    if st.button("✨ Optimize All Images", type="primary", key="bulk_optimize_button"):
        if not api_key: st.warning("🚨 Enter OpenAI API key in sidebar.")
        elif not keyword: st.warning("🎯 Enter a primary Keyword in the sidebar.")
        elif not image_sources_input: st.info("➕ Upload images or provide URLs to begin.")
        elif total_images > 20: st.error(f"❌ Max 20 images ({total_images} provided).")
        else:
            st.session_state.bulk_processed_data = []
            st.session_state.bulk_compare_index = None
            client = OpenAI(api_key=api_key)
            st.header("⏳ Processing Images...")
            progress_bar = st.progress(0, text="Initializing...")
            start_time = time.time()
            temp_processed_data, temp_skipped_files, optimized_filenames_set = [], [], set()
            for idx, source in enumerate(image_sources_input):
                progress_text = f"Processing image {idx + 1}/{total_images}...";
                progress_bar.progress((idx + 1)/total_images, text=progress_text)
                result = process_image(source, client, config, optimized_filenames_set)
                if result['status'] == 'success':
                    optimized_filenames_set.add(result['optimized_filename'])
                    temp_processed_data.append(result)
                    if result.get('api_error'): st.warning(f"⚠️ OpenAI issue for {result['original_filename']}: {result['api_error']}. Defaults used.")
                else:
                    temp_skipped_files.append(f"{result['original_filename']} ({result['message']})")
                time.sleep(0.7)
            progress_bar.empty(); end_time = time.time()
            st.session_state.bulk_processed_data = temp_processed_data
            st.success(f"✅ Optimization complete: {len(temp_processed_data)} items processed in {end_time - start_time:.2f}s!")
            if temp_skipped_files:
                st.error("⚠️ Some files were skipped due to errors:")
                for file_info in temp_skipped_files: st.warning(f"- {file_info}")
            st.rerun()

    if st.session_state.bulk_processed_data:
        st.header("📊 Results & Downloads")
        display_df_data = [{"Optimized Filename": item["optimized_filename"], "Image Title": item["image_title"], "Alt Text": item["alt_text"]} for item in st.session_state.bulk_processed_data]
        st.dataframe(pd.DataFrame(display_df_data), use_container_width=True)
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv = pd.DataFrame(display_df_data).to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV Summary", csv, 'image_seo_summary.csv', 'text/csv', key='csv_download')
        with col_dl2:
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                for item in st.session_state.bulk_processed_data:
                    zf.writestr(item["optimized_filename"], item["compressed_data"])
            st.download_button("📦 Download Optimized Images (.zip)", zip_buffer.getvalue(), 'optimized_images.zip', 'application/zip', key='zip_download')

        st.header("🔄 Edit, Compare & Re-optimize")
        
        reoptimize_cols = st.columns([3, 1])
        with reoptimize_cols[1]:
            if st.button("🔄 Re-optimize Selected", use_container_width=True):
                selected_options = {i: st.session_state.get(f"reoptimize_option_{i}", "none") for i, _ in enumerate(st.session_state.bulk_processed_data)}
                if all(v == "none" for v in selected_options.values()):
                    st.warning("Please select an optimization action for at least one image.")
                else:
                    client = OpenAI(api_key=api_key)
                    existing_filenames = {item['optimized_filename'] for i, item in enumerate(st.session_state.bulk_processed_data) if selected_options.get(i) not in ["regen_text", "regen_all"]}
                    updated_data = st.session_state.bulk_processed_data.copy()
                    
                    with st.spinner("Re-optimizing selected images..."):
                        for i, item in enumerate(updated_data):
                            option = selected_options.get(i)
                            if option == "none": continue
                            try:
                                image = Image.open(BytesIO(item['original_data']))
                                if option in ["regen_text", "regen_all"]:
                                    base_from_api, alt_text, api_error = _generate_ai_metadata(image, client, config)
                                    item['alt_text'] = alt_text
                                    new_unique_filename = _finalize_filename(base_from_api, config['sanitized_project_number'], existing_filenames)
                                    item['optimized_filename'] = new_unique_filename
                                    item['image_title'] = create_title_from_filename(new_unique_filename) # Regenerate title
                                    existing_filenames.add(new_unique_filename)
                                    if api_error: st.warning(f"OpenAI API Error for {item['original_filename']}: {api_error}")
                                
                                if option in ["recompress", "regen_all"]:
                                    compressed_data, compressed_size = _compress_image(image, config['compression_quality'])
                                    item['compressed_data'] = compressed_data
                                    item['compressed_size_bytes'] = compressed_size
                                    item['savings_percentage'] = ((item['original_size_bytes'] - compressed_size) / item['original_size_bytes']) * 100 if item['original_size_bytes'] > 0 else 0.0
                            except Exception as e:
                                st.error(f"Error re-optimizing {item['original_filename']}: {e}")
                    
                    st.session_state.bulk_processed_data = updated_data
                    st.success("Re-optimization complete!")
                    time.sleep(1)
                    st.rerun()
        
        reoptimize_options_map = {'none': 'None', 'regen_text': 'Re-gen Text', 'recompress': 'Re-compress', 'regen_all': 'Re-gen All'}

        for i_disp, item in enumerate(st.session_state.bulk_processed_data):
            is_expanded = st.session_state.bulk_compare_index == i_disp
            with st.expander(f"Image {i_disp+1}: {item.get('original_filename', 'N/A')}", expanded=is_expanded):
                with st.container(border=True):
                    st.radio(
                        "Re-optimization Action:",
                        options=list(reoptimize_options_map.keys()),
                        key=f"reoptimize_option_{i_disp}",
                        horizontal=True,
                        format_func=reoptimize_options_map.get,
                        index=0
                    )
                    st.markdown("---")
                    
                    col1_exp, col2_exp = st.columns([2, 1])
                    with col1_exp:
                        # Fetch values
                        filename = item.get('optimized_filename', '')
                        image_title = item.get('image_title', '')
                        alt_text = item.get('alt_text', '')

                        # Display fields with character counters
                        st.text_input("Optimized Filename", value=filename, key=f"bulk_fn_{i_disp}", on_change=update_bulk_filename, args=(i_disp,))
                        st.caption(f"Character count: {len(filename)}")
                        
                        st.text_input("Image Title", value=image_title, key=f"bulk_title_{i_disp}", on_change=update_bulk_title, args=(i_disp,))
                        st.caption(f"Character count: {len(image_title)}")
                        
                        st.text_area("Generated Alt Text", value=alt_text, height=100, key=f"bulk_alt_{i_disp}", on_change=update_bulk_alt_text, args=(i_disp,))
                        st.caption(f"Character count: {len(alt_text)}")

                        # Update copy-paste output
                        st.markdown("**Copy-Paste Friendly Output**")
                        copy_paste_text = f"Filename: /{filename}\nAlt text: {alt_text}"
                        st.code(copy_paste_text, language='text')

                    with col2_exp:
                        st.metric(label="Original Size", value=format_size(item['original_size_bytes']))
                        st.metric(label="Compressed Size", value=format_size(item['compressed_size_bytes']), delta=f"-{item['savings_percentage']:.1f}%", delta_color="inverse")
                        st.image(BytesIO(item["compressed_data"]), caption="Compressed Preview")
                    
                    st.markdown("---")
                    
                    if is_expanded:
                        st.subheader(f"Comparing: {item.get('original_filename', 'Original Image')}")
                        try:
                            original_img = Image.open(BytesIO(item["original_data"]))
                            compressed_img = Image.open(BytesIO(item["compressed_data"]))
                            image_comparison(img1=original_img, img2=compressed_img, label1="Original", label2=f"WebP Q{compression_quality}")
                            if st.button("Close Comparison", key=f"close_compare_{i_disp}"):
                                st.session_state.bulk_compare_index = None
                                st.rerun()
                        except Exception as compare_err:
                            st.error(f"Could not load images for comparison: {compare_err}")
                    elif image_comparison_available and item.get('original_data'):
                        if st.button("Compare Original vs. Compressed", key=f"compare_btn_{i_disp}"):
                            st.session_state.bulk_compare_index = i_disp
                            st.rerun()

# --- SINGULAR OPTIMIZER TAB ---
with tab2:
    st.header("🖼️ Single Image Input")
    single_uploaded_file = st.file_uploader("Upload an Image", type=['png','jpg','jpeg','webp','gif','bmp','avif'], accept_multiple_files=False, key="single_uploader")
    single_image_url = st.text_input("Or Enter an Image URL", key="single_url_input")
    single_source = None
    if single_uploaded_file: single_source = single_uploaded_file
    elif single_image_url: single_source = single_image_url.strip()

    if st.button("✨ Optimize Single Image", type="primary", key="single_optimize_button"):
        if not api_key: st.warning("🚨 Enter OpenAI API key in sidebar.")
        elif not keyword: st.warning("🎯 Enter a primary Keyword in the sidebar.")
        elif not single_source: st.info("➕ Upload an image or provide a URL to begin.")
        else:
            st.session_state.single_processed_item = None
            st.session_state.single_compare_active = False
            client = OpenAI(api_key=api_key)
            with st.spinner("Processing your image..."):
                result = process_image(single_source, client, config)
                if result['status'] == 'success':
                    st.session_state.single_processed_item = result
                    st.success("✅ Optimization Complete!")
                    if result.get('api_error'):
                         st.warning(f"⚠️ OpenAI issue: {result['api_error']}. Defaults were used.")
                else:
                    st.error(f"❌ Error: {result['message']}")
            st.rerun()
    
    if st.session_state.get('single_processed_item'):
        item = st.session_state.single_processed_item
        st.header("📊 Result & Download")
        with st.container(border=True):
            # Fetch values
            filename = item.get('optimized_filename', '')
            image_title = item.get('image_title', '')
            alt_text = item.get('alt_text', '')

            # Display fields with character counters
            st.markdown(f"**Original Filename:** `{item['original_filename']}`")
            st.text_input("Optimized Filename", value=filename, key="single_fn_edit", on_change=update_single_filename)
            st.caption(f"Character count: {len(filename)}")
            
            st.text_input("Image Title", value=image_title, key="single_title_edit", on_change=update_single_title)
            st.caption(f"Character count: {len(image_title)}")
            
            st.text_area("Generated Alt Text", value=alt_text, height=100, key="single_alt_edit", on_change=update_single_alt_text)
            st.caption(f"Character count: {len(alt_text)}")

            # Update copy-paste output
            st.markdown("**Copy-Paste Friendly Output**")
            copy_paste_text_single = f"Filename: /{filename}\nAlt text: {alt_text}"
            st.code(copy_paste_text_single, language='text')
            
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric(label="Original Size", value=format_size(item['original_size_bytes']))
            metric_col2.metric(label="Compressed Size", value=format_size(item['compressed_size_bytes']), delta=f"-{item['savings_percentage']:.1f}%", delta_color="inverse")
            
            st.download_button(label="📥 Download Optimized Image (.webp)", data=item['compressed_data'], file_name=item['optimized_filename'], mime='image/webp', key='single_download_button')
            st.divider()

            st.subheader("Image Preview & Comparison")
            if st.session_state.get('single_compare_active'):
                 try:
                    original_img = Image.open(BytesIO(item["original_data"]))
                    compressed_img = Image.open(BytesIO(item["compressed_data"]))
                    image_comparison(img1=original_img, img2=compressed_img, label1="Original", label2=f"WebP Q{compression_quality}")
                    if st.button("Close Comparison", key="single_close_compare"):
                         st.session_state.single_compare_active = False
                         st.rerun()
                 except Exception:
                     st.error("Could not load images for comparison.")
            elif image_comparison_available and item.get('original_data'):
                 st.image(BytesIO(item["compressed_data"]), caption="Compressed Preview")
                 if st.button("Compare Original vs. Compressed", key="single_compare_btn"):
                     st.session_state.single_compare_active = True
                     st.rerun()
            else:
                st.image(BytesIO(item["compressed_data"]), caption="Compressed Preview (Comparison tool not available)")```
