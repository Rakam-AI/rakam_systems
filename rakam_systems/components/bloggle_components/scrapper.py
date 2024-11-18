import logging
import os
import csv
import sys
from typing import Dict, List, Tuple
import dotenv

from sklearn.metrics.pairwise import cosine_similarity
import nltk
from playwright.sync_api import sync_playwright
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import openai

dotenv.load_dotenv()

# Ensure NLTK stopwords are downloaded
nltk.download("stopwords")
stop_words = set(stopwords.words("english")) | set(stopwords.words("french"))
additional_stopwords = {
    "des",
    "les",
    "vous",
    "dans",
    "avec",
    "pour",
    "une",
    "ce",
    "cet",
    "cette",
    "de",
    "et",
    "de",
}
custom_stop_words = stop_words | additional_stopwords

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import your existing components
RAKAM_SYSTEMS_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))  # ingestion  # this file
)
sys.path.append(RAKAM_SYSTEMS_DIR)

from rakam_systems.system_manager import SystemManager
from rakam_systems.components.component import Component

class Scrapper(Component):
    def __init__(self, system_manager: SystemManager, api_key: str) -> None:
        """
        Initializes the Scrapper component.

        :param system_manager: Instance of SystemManager.
        :param api_key: OpenAI API key.
        """
        self.system_manager = system_manager
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=self.api_key)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logging.info("Scrapper initialized")

    def extract_headers_with_levels(self, url: str, level: str = 'h2') -> List[Tuple[str, str]]:
        """
        Extract headers from a given URL using Playwright.

        :param url: The URL of the webpage to scrape.
        :param level: The header level to extract (e.g., 'h2').
        :return: A list of tuples containing header level and text.
        """
        headers = []

        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url)
                page.wait_for_load_state('networkidle', timeout=60000)

                headers = [
                    (level, header.text_content().strip())
                    for header in page.query_selector_all(level)
                ]
            except Exception as e:
                logging.error(f"Failed to extract headers from {url}: {e}")
            finally:
                browser.close()

        return headers

    def gather_outlines_from_links(self, links_dict: Dict[str, str]) -> List[str]:
        """
        Gather outlines from each URL in links_dict.

        :param links_dict: Dictionary mapping names to URLs.
        :return: List of outline texts.
        """
        all_outlines = []
        for name, url in links_dict.items():
            logging.info(f"Extracting outlines from: {name} - {url}")
            headers = self.extract_headers_with_levels(url)

            outline_text = ""
            for level, content in headers:
                outline_text += f"{content}\n"

            all_outlines.append(outline_text)
            logging.info(f"Extracted outline from {name}: {outline_text[:100]}...")  # Preview
        return all_outlines

    def generate_prompts_from_text_prompt(self, prompt: str, model: str = "gpt-4") -> str:
        """
        Send a prompt to the OpenAI API and retrieve the response.

        :param prompt: The prompt to send to the API.
        :param model: The OpenAI model to use.
        :return: The response from the API.
        """
        try:
            completion = self.client.chat.completions.create(
                temperature=1,
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error generating prompt response: {e}")
            return ""

    def generate_outlines_from_context(self, outlines_list: List[str]) -> str:
        """
        Generate a cohesive outline list based on a given list of outlines.

        :param outlines_list: List of outline texts.
        :return: Generated cohesive outline.
        """
        outlines_text = "\n\n".join(outlines_list)
        prompt = (
            "Write an outline list based on the context of the provided outlines. "
            "Use only the information provided to create a coherent and unique outline. "
            "This is the list of suggested outlines:\n\n" + outlines_text
        )
        return self.generate_prompts_from_text_prompt(prompt)

    def generate_article_title(self, outline: str) -> str:
        """
        Generate an article title based on the generated outline.

        :param outline: The cohesive outline.
        :return: Suggested article title.
        """
        prompt = (
            "Based on the following outline, suggest a catchy, SEO-optimized article title that captures the main theme "
            "and purpose of the content:\n\n" + outline
        )
        return self.generate_prompts_from_text_prompt(prompt)

    def compute_similarity(self, generated_title: str, existing_titles: List[str]) -> List[Tuple[str, float]]:
        """
        Compute the cosine similarity between the generated title and existing titles.

        :param generated_title: The title generated by the model.
        :param existing_titles: List of existing titles to compare against.
        :return: List of tuples containing existing titles and their similarity scores.
        """
        generated_title_embedding = self.embedding_model.encode(generated_title).reshape(1, -1)
        existing_titles_embeddings = self.embedding_model.encode(existing_titles)

        similarities = cosine_similarity(generated_title_embedding, existing_titles_embeddings).flatten()
        title_similarity_pairs = list(zip(existing_titles, similarities))
        title_similarity_pairs = sorted(title_similarity_pairs, key=lambda x: x[1], reverse=True)

        return title_similarity_pairs

    def save_results_to_csv(
        self,
        filename: str,
        generated_outline: str,
        generated_title: str,
        similarity_rankings: List[Tuple[str, float]]
    ) -> None:
        """
        Save the generated outline, title, and similarity rankings to a CSV file.

        :param filename: Name of the CSV file to save the results.
        :param generated_outline: The cohesive outline generated from existing outlines.
        :param generated_title: The suggested title for the article.
        :param similarity_rankings: List of tuples containing existing titles and their similarity scores.
        """
        with open(filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            
            # Write the generated outline
            writer.writerow(["Generated Outline"])
            writer.writerow([generated_outline])
            writer.writerow([])  # Blank line for separation
            
            # Write the generated title
            writer.writerow(["Generated Title"])
            writer.writerow([generated_title])
            writer.writerow([])  # Blank line for separation
            
            # Write similarity rankings
            writer.writerow(["Existing Title", "Similarity Score"])
            for title, score in similarity_rankings:
                writer.writerow([title, f"{score:.4f}"])
        
        logging.info(f"Results saved to {filename}")

    def process_links(self, links_dict: Dict[str, str], output_csv: str = "generated_article_results.csv") -> None:
        """
        Main method to process the links and generate the required outputs.

        :param links_dict: Dictionary mapping names to URLs.
        :param output_csv: Filename for the CSV output.
        """
        # Gather outlines from each URL
        retrieved_outlines = self.gather_outlines_from_links(links_dict)
        
        # Generate cohesive outline based on retrieved outlines
        cohesive_outline = self.generate_outlines_from_context(retrieved_outlines)
        logging.info(f"Generated Outline:\n{cohesive_outline}")
        
        # Generate article title based on the cohesive outline
        article_title = self.generate_article_title(cohesive_outline)
        logging.info(f"Suggested Article Title:\n{article_title}")
        
        # Get existing titles from the dictionary keys
        existing_titles = list(links_dict.keys())
        
        # Compute similarity rankings
        similarity_rankings = self.compute_similarity(article_title, existing_titles)
        logging.info("Similarity Rankings with Existing Titles:")
        for title, score in similarity_rankings:
            logging.info(f"Title: {title}, Similarity Score: {score:.4f}")
        
        # Save results to CSV
        self.save_results_to_csv(output_csv, cohesive_outline, article_title, similarity_rankings)

    def call_main(self, **kwargs) -> Dict:
        return super().call_main(**kwargs)
    
    def test(self, **kwargs) -> bool:
        return super().test(**kwargs)

if __name__ == "__main__":
    system_manager = SystemManager(system_config_path="system_config.yaml")

    links_dict = {
        "Blogging on Shopify (Complete Guide + Limitations Explained)": "https://bloggle.app/blog/blogging-on-shopify-complete-guide",
        "Blog d'entreprise : avantages et exemples": "https://www.shopify.com/fr/blog/site-e-commerce-blogging",
        "Blogging on Shopify: How to Do It, Examples, Best Practices": "https://greenflagdigital.com/blogging-on-shopify/",
        "Add a blog to your online store": "https://help.shopify.com/en/manual/online-store/blogs/adding-a-blog",
        "Blogs": "https://help.shopify.com/en/manual/online-store/blogs",
        "How Eastside Golf Redefines the Sport With Contemporary Design and Inclusivity": "https://www.shopify.com/blog/eastside-golf-design-and-inclusivity",
        "Blogging on Shopify: How To Do It For SEO Traffic": "https://logeix.com/shopify-seo/blogging"
    }

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    scrapper = Scrapper(system_manager, api_key=OPENAI_API_KEY)

    scrapper.process_links(links_dict, output_csv="generated_article_results.csv")
