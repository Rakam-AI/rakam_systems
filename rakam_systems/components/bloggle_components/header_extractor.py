import logging
from playwright.sync_api import sync_playwright
from rakam_systems.system_manager import SystemManager
from rakam_systems.components.component import Component

class HeaderExtractor(Component):
    def __init__(self, system_manager: SystemManager = None) -> None:
        """
        Initializes the HeaderExtractor component.
        """
        self.system_manager = system_manager
        logging.info("HeaderExtractor initialized")

    def extract_headers(self, url: str) -> list:
        """
        Extracts headers (H1 to H3) from the given URL.
        """
        headers = []
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                page.goto(url)
                page.wait_for_load_state('networkidle', timeout=60000)
                for level in range(1, 4):
                    headers.extend(
                        [(f'h{level}', header.text_content().strip()) 
                         for header in page.query_selector_all(f'h{level}') if header.text_content().strip()]
                    )
            except Exception as e:
                logging.error(f"Error extracting headers: {e}")
            finally:
                browser.close()
        return headers

    def call_main(self, **kwargs) -> dict:
        return super().call_main(**kwargs)
    
    def test(self, **kwargs) -> bool:
        return super().test(**kwargs)
    
if __name__ == "__main__":
    header_extractor = HeaderExtractor()
    headers = header_extractor.extract_headers("https://www.britannica.com/technology/artificial-intelligence")
    print(headers)