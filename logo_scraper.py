from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # plain tqdm, not tqdm_asyncio
import pandas as pd
import requests
from requests import exceptions as rex
from selectolax.parser import HTMLParser
from urllib.parse import urlparse, urljoin
import os
import time
import os, sys
import threading

from collections import Counter

# ----------------------------------------------------------
# Global stats tracked for summary
# ----------------------------------------------------------
stats = Counter()
_thread_local = threading.local()
MAX_WORKERS = 100          # tune per bandwidth / memory

# ----------------------------------------------------------
# Configuration constants
# ----------------------------------------------------------

OUTPUT_CSV = "data/output_pool.csv"
HTML_DEBUG_DIR = "data/debug/html_debug"
BASEURL_LOG_FILE = "data/debug/baseurl_debug.log"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Connection": "keep-alive",
    "Sec-Fetch-Mode": "navigate",
    "Upgrade-Insecure-Requests": "1",
}

HEADERS2 = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "accept-encoding": "gzip, deflate, br, zstd",
    "accept-language": "en-GB,en-US;q=0.9,en;q=0.8,ro;q=0.7,de;q=0.6,nl;q=0.5",
    "cache-control": "max-age=0",
    "cookie": "zcid=AAAAAIVRkuQ-d5SPQ9bO4rh_WrlYESssc... (trimmed)",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.6834.210 Safari/537.36",
    "sec-ch-ua": "\"Not A(Brand\";v=\"8\", \"Chromium\";v=\"132\", \"Opera\";v=\"117\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\"",
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1"
}


def is_valid_svg_content(content: str) -> bool:
    return '<svg' in content.lower() and '<?xml' in content.lower()[:100]

def is_valid_ico_content(content: bytes) -> bool:
    # Check ICO file magic number: first 2 bytes should be 0, next 2 should be 1
    return content[:4] == b'\x00\x00\x01\x00'


def detailed_log_baseurl_attempt(
    domain,
    variant,
    status_code=None,
    final_url=None,
    success=False,
    error=None,
    html_snippet=None,
    redirect_chain=None,
    notes=None,
):
    os.makedirs("data", exist_ok=True)
    #timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    log_entry = (
        #f"[{timestamp}] Domain: {domain}\n"
        f"  ↳ Attempted URL: {variant}\n"
        f"  ↳ Status: {status_code if status_code else '❌ Error'}\n"
        f"  ↳ Final URL: {final_url or 'N/A'}\n"
        f"  ↳ Result: {'✅ Success' if success else '❌ Fail'}\n"
    )

    if redirect_chain:
        log_entry += "  ↳ Redirect chain:\n"
        for i, url in enumerate(redirect_chain):
            log_entry += f"     {i+1}. {url}\n"

    if notes:
        log_entry += f"  ↳ Notes: {notes}\n"

    if error:
        log_entry += f"  ↳ Exception: {str(error)}\n"

    if html_snippet:
        snippet_preview = html_snippet[:200].replace("\n", " ").strip()
        log_entry += f"  ↳ HTML snippet: {snippet_preview}...\n"

    log_entry += "[-----------------------------------]\n\n"

    with open(BASEURL_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)

# ----------------------------------------------------------
# Core function to find a working base URL (www vs no www, http vs https)
# ----------------------------------------------------------

def detect_base_url(domain: str,session: requests.Session) -> tuple[str | None, str | None]:
    # Step 1: Define all common domain variants to try
    variants = [
        f"https://www.{domain}",
        f"https://{domain}",
        f"http://{domain}",
        f"http://www.{domain}",
    ]

    for variant in variants:
        redirect_chain = []
        try:
            
            # Step 2: Make a request to the variant with 10s timeout and follow redirects
            #r =  requests.get(variant, timeout=10, allow_redirects=True,headers=HEADERS)
            #r = session.get(variant, timeout=(5, 10), allow_redirects=True)
            t0 = time.perf_counter()
            #r = session.get(variant, timeout=(3, 7), allow_redirects=True)
            
            #lets try this 
            head = session.head(variant, timeout=(3,7), allow_redirects=True)
            if head.status_code >= 400: continue
            r = session.get(str(head.url), timeout=(3,7))   

            #trying to figure out 
            elapsed = time.perf_counter() - t0
            if elapsed > 8:
                print(f"⚠ URL variant {variant} took {elapsed:.1f}s")

            # Step 3: Save the full redirect chain for later logging/debugging
            redirect_chain = [str(resp.url) for resp in r.history] + [str(r.url)]

            
            # Step 4: Analyze HTML content
            html_text = str(r.content)
            
            is_html_like = "<html" in html_text  # basic check to see if HTML is present
            
            is_spa_suspect = (r.status_code == 200 and len(html_text) < 50)  # suspiciously small page

            # Optional notes for logging
            notes = []
            if is_spa_suspect:
                notes.append("Possible SPA (Single Page Application) or heavily client-rendered site")

            # Step 5: Logging 
            detailed_log_baseurl_attempt(
                domain,
                variant,
                status_code=r.status_code,
                final_url=str(r.url),
                success=is_html_like and r.status_code == 200,
                html_snippet=r.text if not is_html_like else None,
                redirect_chain=redirect_chain if len(redirect_chain) > 1 else None,
                notes=", ".join(notes) if notes else None
            )

            # Step 6: If HTML is valid, return the final resolved URL
            if is_html_like and r.status_code == 200:
                return str(r.url), None
            elif is_html_like and r.status_code != 200:
                return None, "site not found/not working"
            elif is_spa_suspect:
                return str(r.url), "spa_suspected"

        # Step 7: Handle known connection issues with logging
        except rex.ConnectTimeout as e:
            detailed_log_baseurl_attempt(domain, variant, error=e, notes="Connection timeout")
        except rex.ReadTimeout as e:
            detailed_log_baseurl_attempt(domain, variant, error=e, notes="Read timeout")
        except rex.RequestException as e: # base-class for HTTPError, SSLError, etc
            detailed_log_baseurl_attempt(domain, variant, error=e, notes="HTTP error")
        except UnicodeDecodeError as e:
            detailed_log_baseurl_attempt(domain, variant, error=e, notes="Unicode decode error")
        except Exception as e:
            detailed_log_baseurl_attempt(domain, variant, error=e, notes="Unhandled exception")
        

    # Step 8: If no variant worked, return None
    detailed_log_baseurl_attempt(domain, "All variants exhausted", notes="No working URL found")
    return None, "no_working_variant"

# ----------------------------------------------------------
# Extract potential logo from HTML
# ----------------------------------------------------------

def extract_logo_url(html, final_url):
    tree = HTMLParser(html)

    def resolve(src):
        if not src:
            return None
        return src if src.startswith("http") else urljoin(final_url, src)

    # 1) Favicon check
    icon = tree.css_first("link[rel*='icon']")
    if icon and icon.attributes.get("href"):
        return "favicon", resolve(icon.attributes["href"])

    # 2) OpenGraph check
    meta = tree.css_first("meta[property='og:image'], meta[name='og:image']")
    if meta and meta.attributes.get("content"):
        return "meta-og", resolve(meta.attributes["content"])

    # 3) Direct <img> with 'logo' in src/class/id
    img = tree.css_first("img[src*='logo'], img[class*='logo'], img[id*='logo']")
    if img and img.attributes.get("src"):
        return "img-logo", resolve(img.attributes["src"])

    # 4) <header> fallback
    header_img = tree.css_first("header img")
    if header_img and header_img.attributes.get("src"):
        return "header-img", resolve(header_img.attributes["src"])

    # 5) Extra fallback: any <img> containing .svg or 'brand' or 'header'
    fallback = tree.css_first("img[src*='.svg'], img[src*='brand'], img[src*='header']")
    if fallback and fallback.attributes.get("src"):
        return "svg-like", resolve(fallback.attributes["src"])

    return None, None


def get_session() -> requests.Session:
    if not hasattr(_thread_local, "sess"):
        s = requests.Session() # new session , per worker
        adapter = requests.adapters.HTTPAdapter(pool_connections=MAX_WORKERS,
                                                pool_maxsize=MAX_WORKERS) # the adaptor 
        s.mount("http://", adapter)             # attach to both schemes
        s.mount("https://", adapter)
        s.headers.update(HEADERS)          # default headers once
        _thread_local.sess = s
    return _thread_local.sess


# ----------------------------------------------------------
# Scrape logic for a single domain
# ----------------------------------------------------------
def scrape_domain(domain):
    
    result = {
        "domain": domain,
        "base_url": None,
        "logo_url": None,
        "status": "fail",
        "error": None,
    }
    try:
        session = get_session() ### gettign a session per worker
        base_url, reason =  detect_base_url(domain,session)
        result["base_url"] = base_url
        
        if not base_url:
            stats["base_url_not_found"] += 1
            if reason == "spa_suspected":
                stats["spa_suspected"] += 1
                return result
        else:
            stats["base_url_found"] += 1

        # Fetch final page HTML
        #r =  requests.get(base_url, timeout=10,headers=HEADERS)
        r = session.get(base_url, timeout=(5, 10))
        result["html_length"] = len(r.text)

        # Attempt to extract a logo
        final_url = str(r.url)  # r.url is a yarl.URL object in httpx
        logo_type, logo_url = extract_logo_url(r.text, final_url)
        
        if logo_url:
            stats["logo_found"] += 1
            result["logo_url"] = logo_url
            result["status"] = "success"
        else:
            # No logo found, store HTML for debugging
            stats["logo_not_found"] += 1
            result["status"] = "no_logo_found"
            # Save HTML snapshot for debugging
            #--- Build a filename that matches the domain  (e.g.  example_com.html)
            parsed = urlparse(base_url)
            filename = parsed.netloc.replace(".", "_") + ".html"
            os.makedirs(HTML_DEBUG_DIR, exist_ok=True)
            #os.remove(os.path.join(HTML_DEBUG_DIR, filename))
            path = os.path.join(HTML_DEBUG_DIR, filename)
            #--- Ensure debug directory exists
            #--- Delete any previous snapshot of that site (the guard should go here)
            if os.path.exists(path):          # ← prevent FileNotFoundError
                os.remove(path)

            #--- Write the full HTML page to disk so to inspect manually later
            with open(os.path.join(HTML_DEBUG_DIR, filename), "w", encoding="utf-8") as f:
                f.write(r.text)

    except Exception as e:
        # Catch unexpected exceptions
        stats["exception"] += 1
        result["error"] = str(e)
        result["status"] = "exception"

    return result

# ----------------------------------------------------------
# Main async driver
# ----------------------------------------------------------


def main():
    
    print(f"PID {os.getpid()}  –  {sys.executable}")
    sys.stdout.flush()        # make sure it appears immediately

    t0 = time.perf_counter()  
    df = pd.read_parquet("data/logos_snappy.parquet", engine="pyarrow")
    domains = df["domain"].dropna().unique().tolist()[:500] # Limit to first 200 for testing

    results, stats = [], Counter()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(scrape_domain, d): d for d in domains}

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scraping"):
            try:
                res = fut.result(timeout=20)
                results.append(res)
            
            except TimeoutError:
                # task still running after 20 s → treat as failure
                res = {"domain": futures[fut], "status": "timeout"}
                results.append(res)

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"\n✔ Results written to {OUTPUT_CSV}")
    
    elapsed = time.perf_counter() - t0        # ◄──┘ stop
    print(f"\n⏱  Finished {len(domains)} domains in {elapsed:,.2f} s "
          f"({len(domains)/elapsed:,.1f} req/s)")

    print("\n🧾 Summary counters:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    


if __name__ == "__main__":
    (main())