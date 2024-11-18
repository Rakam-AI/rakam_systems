import logging
from rakam_systems.system_manager import SystemManager
from rakam_systems.components.component import Component
from playwright.sync_api import sync_playwright
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class GoogleSearcher(Component):
    def __init__(self, system_manager: SystemManager = None) -> None:
        """
        Initializes the GoogleSearcher component.

        :param system_manager: Instance of SystemManager.
        """
        self.system_manager = system_manager
        self.number_of_keywords = 300
        logging.info("GoogleSearcher initialized")

    def google_search(self, keyword: str, language: str = 'en', country: str = 'us') -> dict:
        """
        Performs a Google search and returns the top 10 results.

        :param keyword: The search query.
        :param language: Language code for search results.
        :param country: Country code for search results.
        :return: Dictionary of titles and links.
        """
        search_url = f"https://www.google.com/search?q={keyword}&hl={language}&gl={country}"
        results = {}

        logging.info(f"Performing search for keyword: {keyword}")
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
            )
            page = context.new_page()

            try:
                page.goto(search_url)
                page.wait_for_selector("h3")

                # Extract the titles and links from the search results
                search_results = page.query_selector_all("a:has(h3)")
                for result in search_results[:10]:  # Get top 10 results
                    link = result.get_attribute("href")
                    title = result.query_selector("h3").inner_text() if result.query_selector("h3") else "No title"
                    if link and title:
                        results[title] = link

                logging.info(f"Search completed for keyword: {keyword}")
            except Exception as e:
                logging.error(f"Error during search: {e}")
            finally:
                browser.close()

        return results

    def batch_search(self, keywords: list, language: str = 'en', country: str = 'us') -> dict:
        """
        Performs a Google search for multiple keywords.

        :param keywords: List of search queries.
        :param language: Language code for search results.
        :param country: Country code for search results.
        :return: Dictionary with keywords as keys and their search results.
        """
        all_results = {}
        for keyword in keywords[:self.number_of_keywords]:
            all_results[keyword] = self.google_search(keyword, language, country)
        return all_results
    
    def extract_keywords(self, text: str, n: int = 300) -> list:
        """
        Extracts top n keywords using CountVectorizer.
        """
        vectorizer = CountVectorizer(ngram_range=(1, 2))
        X = vectorizer.fit_transform([text])
        terms = vectorizer.get_feature_names_out()
        counts = X.sum(axis=0).A1
        term_counts = list(zip(terms, counts))
        return sorted(term_counts, key=lambda x: x[1], reverse=True)[:n]

    def call_main(self, **kwargs) -> dict:
        return super().call_main(**kwargs)
    
    def test(self, **kwargs) -> bool:
        return super().test(**kwargs)
    
if __name__ == "__main__":
    google_searcher = GoogleSearcher()
    keywords = ["data science", "machine learning", "artificial intelligence"]
    results = google_searcher.batch_search(keywords)
    for keyword, search_results in results.items():
        logging.info(f"Keyword: {keyword}")
        print(f"Keyword: {keyword}")
        for title, link in search_results.items():
            logging.info(f"{title}: {link}")
            print(f"{title}: {link}")
        logging.info("\n")
        print("\n")
        print("=" * 50)