#!/usr/bin/env python3
"""
ArXiv Daily Summary Generator
Fetches papers from arXiv, filters by user preferences, and generates summaries using Ollama.
"""

import arxiv
import requests
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Union
import time
import markdown
from weasyprint import HTML
import tempfile
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import sys

from pydantic import BaseModel, ValidationError


from .config import EmailSettings, MattermostSettings, SummarizerSettings, load_yaml_config

try:
    from mattermost_api_reference_client import AuthenticatedClient
    from mattermost_api_reference_client.api.posts import create_post
    from mattermost_api_reference_client.models.create_post_body import CreatePostBody
    MATTERMOST_AVAILABLE = True
except ImportError as e:
    print(f"Warning: mattermost-api-reference-client not installed. Mattermost functionality will be disabled. ({e})")
    MATTERMOST_AVAILABLE = False


# Category configurations with default preferences
CATEGORY_CONFIGS = {
    'astro-ph.CO': {
        'name': 'Cosmology and Nongalactic Astrophysics',
        'default_preferences': [
            'dark energy',
            'cosmic microwave background',
            'large scale structure',
            'galaxy clusters',
            'cosmological simulations'
        ]
    },
    'astro-ph': {
        'name': 'Astrophysics',
        'default_preferences': [
            'dark energy',
            'cosmic microwave background',
            'large scale structure',
            'galaxy clusters',
            'cosmological simulations'
        ]
    },
    'stat.ML': {
        'name': 'Machine Learning (Statistics)',
        'default_preferences': [
            'deep learning',
            'neural networks',
            'generative models',
            'reinforcement learning',
            'optimization',
            'causal inference'
        ]
    },
    'cs.LG': {
        'name': 'Machine Learning (Computer Science)',
        'default_preferences': [
            'deep learning',
            'neural networks',
            'generative models',
            'reinforcement learning',
            'optimization',
            'transformers'
        ]
    },
    'physics': {
        'name': 'Physics (All Categories)',
        'default_preferences': [
            'quantum mechanics',
            'condensed matter',
            'statistical physics',
            'particle physics',
            'quantum field theory'
        ],
        'query': 'cat:physics.*'  # Special query for all physics
    },
    'physics.gen-ph': {
        'name': 'General Physics',
        'default_preferences': [
            'fundamental physics',
            'theoretical physics',
            'experimental physics'
        ]
    },
    'hep-th': {
        'name': 'High Energy Physics - Theory',
        'default_preferences': [
            'string theory',
            'quantum field theory',
            'gauge theory',
            'supersymmetry'
        ]
    },
    'cond-mat': {
        'name': 'Condensed Matter (All)',
        'default_preferences': [
            'quantum materials',
            'superconductivity',
            'topological phases',
            'phase transitions'
        ],
        'query': 'cat:cond-mat.*'
    },
    'quant-ph': {
        'name': 'Quantum Physics',
        'default_preferences': [
            'quantum computing',
            'quantum information',
            'quantum entanglement',
            'quantum algorithms'
        ]
    }
}


class MattermostBot:
    """Mattermost bot for sending messages."""

    def __init__(self, server_url: str, bot_token: str, channel_id: str):
        """
        Initialize Mattermost bot.

        Args:
            server_url: Mattermost server URL (e.g., https://mattermost.example.com)
            bot_token: Bot access token
            channel_id: Channel ID where to post messages
        """
        self.server_url = server_url.rstrip('/')
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.client = None

    def connect(self) -> bool:
        """
        Connect to Mattermost server.

        Returns:
            True if connection successful, False otherwise
        """
        if not MATTERMOST_AVAILABLE:
            print("Error: mattermost-api-reference-client is not installed. Install with: pip install mattermost-api-reference-client")
            return False

        try:
            # Initialize authenticated client
            self.client = AuthenticatedClient(
                base_url=self.server_url,
                token=self.bot_token
            )

            print(f"Successfully connected to Mattermost: {self.server_url}")
            return True
        except Exception as e:
            print(f"Error connecting to Mattermost: {e}")
            print(f"  Server URL: {self.server_url}")
            print(f"  Make sure the server URL is valid (e.g., https://mattermost.example.com)")
            return False

    def send_message(self, message: str, file_ids: Optional[List[str]] = None) -> bool:
        """
        Send a message to the configured channel.

        Args:
            message: Message to send
            file_ids: Optional Mattermost file IDs to attach

        Returns:
            True if message sent successfully, False otherwise
        """
        if not self.client:
            print("Error: Not connected to Mattermost. Call connect() first.")
            return False

        try:
            # Mattermost rejects empty messages; provide a short placeholder when only a file is posted.
            final_message = message if message.strip() else "Attached file"
            post_body = CreatePostBody(channel_id=self.channel_id, message=final_message, file_ids=file_ids)

            # Send message using the API
            create_post.sync(body=post_body, client=self.client)

            print("Message sent to Mattermost successfully")
            return True
        except Exception as e:
            print(f"Error sending message to Mattermost: {e}")
            return False

    def upload_file(self, file_path: str) -> Optional[str]:
        """Upload a file to Mattermost and return the file ID."""
        if not Path(file_path).exists():
            print(f"File not found for Mattermost upload: {file_path}")
            return None

        upload_url = f"{self.server_url}/api/v4/files"
        headers = {"Authorization": f"Bearer {self.bot_token}"}

        try:
            with open(file_path, "rb") as file_handle:
                files = {"files": (Path(file_path).name, file_handle, "application/pdf")}
                data = {"channel_id": self.channel_id}
                response = requests.post(upload_url, headers=headers, data=data, files=files, timeout=120)
                response.raise_for_status()

            file_infos = response.json().get("file_infos", [])
            if not file_infos:
                print(f"Mattermost did not return file info for upload of {file_path}")
                return None

            file_id = file_infos[0].get("id")
            print(f"Uploaded file to Mattermost: {file_path} (id={file_id})")
            return file_id
        except Exception as e:
            print(f"Error uploading file to Mattermost: {e}")
            return None

    def send_file(self, file_path: str, message: str = "") -> bool:
        """Upload a file and post it to the channel."""
        file_id = self.upload_file(file_path)
        if not file_id:
            return False

        return self.send_message(message, file_ids=[file_id])

    def disconnect(self):
        """Disconnect from Mattermost server."""
        if self.client:
            self.client = None
            print("Disconnected from Mattermost")


class ArxivSummarizer:
    @staticmethod
    def _config_to_dict(config_obj: Optional[Union[BaseModel, Dict]]) -> Dict:
        """Normalize config objects (Pydantic or dict) into plain dicts."""
        if config_obj is None:
            return {}
        if isinstance(config_obj, BaseModel):
            return config_obj.model_dump(exclude_none=True)
        if isinstance(config_obj, dict):
            return {k: v for k, v in config_obj.items() if v is not None}
        return {}

    def __init__(self, preferences_file: str = "preferences.txt",
                 ollama_url: str = "http://localhost:11434",
                 model: str = "llama3.2",
                 auth_token: Optional[str] = None,
                 category: str = "astro-ph.CO",
                 email_config: Optional[Union[EmailSettings, Dict]] = None,
                 mattermost_config: Optional[Union[MattermostSettings, Dict]] = None):
        """
        Initialize the ArXiv summarizer.

        Args:
            preferences_file: Path to file containing preferred topics (one per line)
            ollama_url: URL for Ollama API
            model: Ollama model to use for summarization
            auth_token: Bearer token for authentication (or set OLLAMA_AUTH_TOKEN env var)
            category: arXiv category code
            email_config: Email configuration dict with keys:
                - smtp_server: SMTP server address
                - smtp_port: SMTP port (default: 587)
                - sender_email: Sender email address
                - sender_password: Sender email password (or use EMAIL_PASSWORD env var)
                - recipient_email: Recipient email address
            mattermost_config: Mattermost configuration dict with keys:
                - server_url: Mattermost server URL
                - bot_token: Bot access token (or use MATTERMOST_BOT_TOKEN env var)
                - channel_id: Channel ID to post messages
        """
        self.preferences_file = Path(preferences_file)
        self.ollama_url = ollama_url
        self.model = model
        # Check for auth token in env variable if not provided
        self.auth_token = auth_token or os.getenv('OLLAMA_AUTH_TOKEN')
        self.category = category
        self.category_config = CATEGORY_CONFIGS.get(category, {
            'name': category,
            'default_preferences': []
        })
        self.email_config = self._config_to_dict(email_config)

        # Initialize Mattermost bot
        self.mattermost_bot = None
        self.mattermost_config = self._config_to_dict(mattermost_config)
        if self.mattermost_config:
            bot_token = self.mattermost_config.get('bot_token') or os.getenv('MATTERMOST_BOT_TOKEN')
            if bot_token and self.mattermost_config.get('server_url') and self.mattermost_config.get('channel_id'):
                self.mattermost_config['bot_token'] = bot_token
                self.mattermost_bot = MattermostBot(
                    server_url=self.mattermost_config['server_url'],
                    bot_token=bot_token,
                    channel_id=self.mattermost_config['channel_id']
                )

                assert self.mattermost_bot.connect()
                # self.mattermost_bot.send_message("ArXiv Summarizer bot connected and ready to send messages.")

        self.preferences = self._load_preferences()

    def _load_preferences(self) -> List[str]:
        """Load user preferences from file."""
        if not self.preferences_file.exists():
            print(f"Creating default preferences file: {self.preferences_file}")
            default_prefs = self.category_config.get('default_preferences', [])
            if not default_prefs:
                default_prefs = ['research', 'theory', 'experiment']
            self.preferences_file.write_text("\n".join(default_prefs))
            return default_prefs

        with open(self.preferences_file, 'r') as f:
            prefs = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return prefs

    def fetch_papers(self, category: str = None, days_back: int = 1, max_results: int = 100) -> List[Dict]:
        """
        Fetch papers from arXiv for the specified category.

        Args:
            category: arXiv category (overrides instance category if provided)
            days_back: Number of days to look back
            max_results: Maximum number of papers to fetch

        Returns:
            List of paper dictionaries with metadata
        """
        if category is None:
            category = self.category

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Get query string (some categories use wildcards)
        startDateStr = "{:04d}{:02d}{:02d}0000".format(start_date.year, start_date.month, start_date.day)
        endDateStr = "{:04d}{:02d}{:02d}0000".format(end_date.year, end_date.month, end_date.day)
        query = "{} AND submittedDate:[{} TO {}]".format(self.category_config.get('query', f"cat:{category}"), startDateStr, endDateStr)
        print(query)
        category_name = self.category_config.get('name', category)

        print(f"Fetching papers from {category_name} ({category})")
        print(f"  Date range: {start_date.date()} to {end_date.date()}")

        # Create client and build search query
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )

        papers = []
        first = True
        for result in client.results(search):
            # Filter by submission date
            if first:
              print(f"First paper date: {result.published}")
#              first = False
            if result.published.replace(tzinfo=None) < start_date:
                continue

            paper = {
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'abstract': result.summary,
                'published': result.published.strftime('%Y-%m-%d'),
                'arxiv_id': result.entry_id.split('/')[-1],
                'categories': result.categories,
                'pdf_url': result.pdf_url
            }
            papers.append(paper)

        print(f"Found {len(papers)} papers")
        return papers

    def _calculate_relevance_score(self, paper: Dict) -> float:
        """
        Calculate relevance score based on user preferences.

        Args:
            paper: Paper dictionary

        Returns:
            Relevance score (0-1)
        """
        text = f"{paper['title']} {paper['abstract']}".lower()
        matches = sum(1 for pref in self.preferences if pref.lower() in text)
        return matches / len(self.preferences) if self.preferences else 0

    def filter_papers(self, papers: List[Dict], min_relevance: float = 0.1) -> List[Dict]:
        """
        Filter papers based on relevance to user preferences.

        Args:
            papers: List of paper dictionaries
            min_relevance: Minimum relevance score (0-1)

        Returns:
            Filtered and sorted list of papers
        """
        scored_papers = []
        for paper in papers:
            score = self._calculate_relevance_score(paper)
            if score >= min_relevance:
                paper['relevance_score'] = score
                scored_papers.append(paper)

        # Sort by relevance score (descending)
        scored_papers.sort(key=lambda x: x['relevance_score'], reverse=True)

        print(f"Filtered to {len(scored_papers)} relevant papers")
        return scored_papers

    def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama API for text generation.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        # Prepare headers with optional authentication
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            return response.json()['response']
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return f"[Error generating summary: {e}]"

    def _create_paper_summary_prompt(self, paper: Dict) -> str:
        """Create prompt for summarizing a single paper."""
        # Adjust expertise based on category
        if self.category.startswith('stat.') or self.category.startswith('cs.'):
            expertise = "machine learning and statistics"
        elif self.category.startswith('astro'):
            expertise = "astrophysics"
        elif 'physics' in self.category or self.category in ['hep-th', 'cond-mat', 'quant-ph']:
            expertise = "physics"
        else:
            expertise = "the relevant scientific field"

        return f"""You are an expert in {expertise}. Provide a concise, technical summary of the following arXiv paper in 2-3 sentences. Focus on the key findings, methods, and implications.

Title: {paper['title']}

Abstract: {paper['abstract']}

Summary:"""

    def _create_daily_digest_prompt(self, papers_with_summaries: List[Dict]) -> str:
        """Create prompt for generating overall daily digest."""
        user_interests = ", ".join(self.preferences)
        category_name = self.category_config.get('name', self.category)

        # Adjust expertise based on category
        if self.category.startswith('stat.') or self.category.startswith('cs.'):
            expertise = "machine learning and statistics"
            field_context = "ML research"
        elif self.category.startswith('astro'):
            expertise = "astrophysics"
            field_context = "astrophysical research"
        elif 'physics' in self.category or self.category in ['hep-th', 'cond-mat', 'quant-ph']:
            expertise = "physics"
            field_context = "physics research"
        else:
            expertise = "the relevant scientific field"
            field_context = "recent research"

        papers_text = "\n\n".join([
            f"Paper {i+1}: {p['title']}\n"
            f"Relevance Score: {p['relevance_score']:.2f}\n"
            f"Summary: {p['summary']}\n"
            f"arXiv ID: {p['arxiv_id']}"
            for i, p in enumerate(papers_with_summaries)
        ])

        return f"""You are an expert in {expertise} preparing a daily research digest. Based on the following paper summaries from arXiv's {category_name} category, create a cohesive overview highlighting the most important developments and trends. You will not invent any papers. You will stay factual and avoid grandiloquent words. Do not overuse bold fonts.

User's research interests: {user_interests}

Papers from today:

{papers_text}

Please provide:
1. A brief overview paragraph identifying major themes and breakthrough results in {field_context}
2. Highlight 2-3 papers of particular significance and explain why they matter
3. Note any emerging trends or connections between papers
4. The URL link to the pdf of the arxiv paper for further reading

Daily Digest:"""

    def summarize_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        Generate summaries for each paper using Ollama.

        Args:
            papers: List of paper dictionaries

        Returns:
            Papers with added 'summary' field
        """
        print(f"Generating summaries for {len(papers)} papers...")

        for i, paper in enumerate(papers):
            print(f"  Summarizing paper {i+1}/{len(papers)}: {paper['title'][:60]}...")
            prompt = self._create_paper_summary_prompt(paper)
            summary = self._call_ollama(prompt)
            paper['summary'] = summary.strip()
            time.sleep(0.5)  # Rate limiting

        return papers

    def generate_daily_digest(self, papers_with_summaries: List[Dict]) -> str:
        """
        Generate an overall daily digest from paper summaries.

        Args:
            papers_with_summaries: List of papers with summaries

        Returns:
            Daily digest text
        """
        print("Generating daily digest...")
        prompt = self._create_daily_digest_prompt(papers_with_summaries)
        digest = self._call_ollama(prompt)
        return digest.strip()

    def run(self, category: str = "astro-ph.CO", days_back: int = 1,
            max_results: int = 100, min_relevance: float = 0.1,
            output_file: str = None, output_format: str = "text",
            send_email: bool = False, send_mattermost: bool = False,
            mattermost_content: str = "summaries") -> str:
        """
        Run the complete pipeline.

        Args:
            category: arXiv category
            days_back: Days to look back
            max_results: Maximum papers to fetch
            min_relevance: Minimum relevance threshold
            output_file: Optional file to save output
            output_format: Output format ('text', 'markdown', or 'pdf')
            send_email: Whether to send the digest via email
            send_mattermost: Whether to send the digest to Mattermost
            mattermost_content: What to send to Mattermost ('summaries' or 'pdf')

        Returns:
            Complete digest text
        """
        print("=" * 80)
        print(f"ArXiv Daily Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print(f"\nUser preferences: {', '.join(self.preferences)}\n")

        # Fetch papers
        papers = self.fetch_papers(category, days_back, max_results)

        if not papers:
            return "No papers found for the specified date range."

        # Filter by relevance
        relevant_papers = self.filter_papers(papers, min_relevance)

        if not relevant_papers:
            return "No papers matched your preferences."

        # Generate individual summaries
        papers_with_summaries = self.summarize_papers(relevant_papers[:10])  # Limit to top 10

        # Generate daily digest
        digest = self.generate_daily_digest(papers_with_summaries)

        # Format output based on requested format
        if output_format == "markdown":
            output = self._format_markdown(digest, papers_with_summaries)
        else:
            output = self._format_output(digest, papers_with_summaries)

        # Save to file if requested
        if output_file:
            if output_format == "pdf":
                self._save_as_pdf(output if output_format == "text" else
                                 self._format_markdown(digest, papers_with_summaries),
                                 output_file)
            else:
                Path(output_file).write_text(output)
            print(f"\nSaved digest to {output_file}")

        # Send email if requested
        if send_email:
            self._send_email(output_file, output_format, content=output if not output_file else None)

        # Send to Mattermost if requested
        if send_mattermost:
            self._send_to_mattermost(
                digest,
                papers_with_summaries,
                output_file=output_file,
                output_format=output_format,
                content_mode=mattermost_content,
            )

        return output

    def _format_output(self, digest: str, papers: List[Dict]) -> str:
        """Format the final output."""
        category_name = self.category_config.get('name', self.category)
        output = []
        output.append("=" * 80)
        output.append(f"ArXiv Daily Digest - {datetime.now().strftime('%Y-%m-%d')}")
        output.append(f"Category: {category_name}")
        output.append("=" * 80)
        output.append(f"\nYour Research Interests: {', '.join(self.preferences)}\n")
        output.append("-" * 80)
        output.append("DAILY OVERVIEW")
        output.append("-" * 80)
        output.append(digest)
        output.append("\n" + "-" * 80)
        output.append("INDIVIDUAL PAPER SUMMARIES")
        output.append("-" * 80)

        for i, paper in enumerate(papers):
            output.append(f"\n{i+1}. {paper['title']}")
            output.append(f"   Authors: {', '.join(paper['authors'][:3])}" +
                         (" et al." if len(paper['authors']) > 3 else ""))
            output.append(f"   arXiv ID: {paper['arxiv_id']}")
            output.append(f"   Published: {paper['published']}")
            output.append(f"   Relevance Score: {paper['relevance_score']:.2f}")
            output.append(f"   PDF: {paper['pdf_url']}")
            output.append(f"\n   Summary: {paper['summary']}\n")

        return "\n".join(output)

    def _format_markdown(self, digest: str, papers: List[Dict]) -> str:
        """Format the output as Markdown."""
        category_name = self.category_config.get('name', self.category)
        output = []
        output.append(f"# ArXiv Daily Digest - {datetime.now().strftime('%Y-%m-%d')}\n")
        output.append(f"**Category:** {category_name}\n")
        output.append(f"**Your Research Interests:** {', '.join(self.preferences)}\n")
        output.append("---\n")
        output.append("## Daily Overview\n")
        output.append(digest + "\n")
        output.append("---\n")
        output.append("## Individual Paper Summaries\n")

        for i, paper in enumerate(papers):
            output.append(f"### {i+1}. {paper['title']}\n")
            output.append(f"**Authors:** {', '.join(paper['authors'][:3])}" +
                         (" et al." if len(paper['authors']) > 3 else "") + "\n")
            output.append(f"**arXiv ID:** [{paper['arxiv_id']}]({paper['pdf_url']})")
            output.append(f" | **Published:** {paper['published']}")
            output.append(f" | **Relevance:** {paper['relevance_score']:.2f}\n")
            output.append(f"**Summary:** {paper['summary']}\n")

        return "\n".join(output)

    def _save_as_pdf(self, markdown_content: str, output_file: str):
        """Convert markdown content to PDF."""
        try:
            # Convert markdown to HTML
            html_content = markdown.markdown(markdown_content, extensions=['extra', 'nl2br'])

            # Add CSS styling for better PDF appearance
            styled_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{
                        font-family: 'Georgia', 'Times New Roman', serif;
                        line-height: 1.6;
                        max-width: 800px;
                        margin: 40px auto;
                        padding: 0 20px;
                        color: #333;
                    }}
                    h1 {{
                        color: #2c3e50;
                        border-bottom: 3px solid #3498db;
                        padding-bottom: 10px;
                    }}
                    h2 {{
                        color: #34495e;
                        border-bottom: 2px solid #95a5a6;
                        padding-bottom: 5px;
                        margin-top: 30px;
                    }}
                    h3 {{
                        color: #16a085;
                        margin-top: 25px;
                    }}
                    hr {{
                        border: none;
                        border-top: 1px solid #bdc3c7;
                        margin: 20px 0;
                    }}
                    strong {{
                        color: #2c3e50;
                    }}
                    a {{
                        color: #3498db;
                        text-decoration: none;
                    }}
                    a:hover {{
                        text-decoration: underline;
                    }}
                    code {{
                        background-color: #f4f4f4;
                        padding: 2px 4px;
                        border-radius: 3px;
                    }}
                </style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """

            # Convert HTML to PDF
            HTML(string=styled_html).write_pdf(output_file)
            print(f"PDF generated successfully: {output_file}")

        except Exception as e:
            print(f"Error generating PDF: {e}")
            print("Falling back to markdown output...")
            # Fallback to markdown if PDF generation fails
            Path(output_file.replace('.pdf', '.md')).write_text(markdown_content)

    def _send_email(self, output_file: str, output_format: str, content: str = None):
        """
        Send the digest via email.

        Args:
            output_file: Path to the output file (if saved)
            output_format: Format of the output ('text', 'markdown', 'pdf')
            content: Text content to send if no file is saved
        """
        if not self.email_config:
            print("Email configuration not provided. Skipping email send.")
            return

        # Get email configuration
        smtp_server = self.email_config.get('smtp_server')
        smtp_port = self.email_config.get('smtp_port', 587)
        sender_email = self.email_config.get('sender_email')
        sender_password = self.email_config.get('sender_password') or os.getenv('EMAIL_PASSWORD')
        recipient_email = self.email_config.get('recipient_email')

        if not all([smtp_server, sender_email, sender_password, recipient_email]):
            print("Incomplete email configuration. Required: smtp_server, sender_email, sender_password, recipient_email")
            return

        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = recipient_email
            msg['Subject'] = f"ArXiv Daily Digest - {datetime.now().strftime('%Y-%m-%d')} - {self.category_config.get('name', self.category)}"

            # Email body
            category_name = self.category_config.get('name', self.category)
            body = f"""
ArXiv Daily Digest
Date: {datetime.now().strftime('%Y-%m-%d')}
Category: {category_name}

Your personalized digest is attached.

Research Interests: {', '.join(self.preferences)}
"""
            msg.attach(MIMEText(body, 'plain'))

            # Attach file if it exists
            if output_file and Path(output_file).exists():
                with open(output_file, 'rb') as f:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename={Path(output_file).name}'
                    )
                    msg.attach(part)
            elif content:
                # Attach content as text if no file
                attachment = MIMEText(content, 'plain')
                attachment.add_header(
                    'Content-Disposition',
                    f'attachment; filename=arxiv_digest_{datetime.now().strftime("%Y%m%d")}.txt'
                )
                msg.attach(attachment)

            # Send email
            print(f"Sending email to {recipient_email}...")
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)

            print(f"Email sent successfully to {recipient_email}")

        except Exception as e:
            print(f"Error sending email: {e}")

    def _send_to_mattermost(
        self,
        digest: str,
        papers: List[Dict],
        output_file: Optional[str],
        output_format: str,
        content_mode: str = "summaries",
    ):
        """
        Send the digest to Mattermost.

        Args:
            digest: Daily digest text
            papers: List of papers with summaries
            output_file: Path to existing output file (used for PDF mode)
            output_format: Output format selected by the user
            content_mode: "summaries" to post text summaries, "pdf" to upload the PDF
        """
        if not self.mattermost_bot:
            print("Mattermost bot not configured. Skipping Mattermost send.")
            return

        try:
            # Connect to Mattermost
            if not self.mattermost_bot.connect():
                print("Failed to connect to Mattermost")
                return

            category_name = self.category_config.get('name', self.category)

            print(f"Sending digest to Mattermost channel ID: {self.mattermost_bot.channel_id}")
            print(f"  Content mode: {content_mode}")
            if content_mode == "pdf":
                # Ensure we have a PDF to upload
                temp_file = None
                pdf_path = None

                if output_format == "pdf" and output_file:
                    pdf_path = output_file
                else:
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    temp_file.close()
                    markdown_content = self._format_markdown(digest, papers)
                    self._save_as_pdf(markdown_content, temp_file.name)
                    pdf_path = temp_file.name

                message = (
                    f"ArXiv Daily Digest ({category_name}) — PDF attached."
                )

                if not self.mattermost_bot.send_file(pdf_path, message):
                    print("Failed to send PDF to Mattermost")

                # Clean up temporary file if we created one
                if temp_file:
                    try:
                        Path(temp_file.name).unlink(missing_ok=True)
                    except Exception:
                        pass
            else:
                # Create a formatted message with digest and summaries
                message_lines = [
                    f"## ArXiv Daily Digest - {datetime.now().strftime('%Y-%m-%d')}",
                    f"**Category:** {category_name}",
                    f"**Research Interests:** {', '.join(self.preferences)}",
                    "",
                    "### Daily Overview",
                    digest,
                ]

                # Add paper summaries
                if not papers is None and len(papers) > 0:
                    message_lines.append("")
                    message_lines.append("### Paper Summaries")
                    for i, paper in enumerate(papers):
                        message_lines.append(f"\n**{i+1}. {paper['title']}**")
                        message_lines.append(f"[arXiv:{paper['arxiv_id']}]({paper['pdf_url']})")
                        message_lines.append(f"*Relevance: {paper['relevance_score']:.2f}*")
                        message_lines.append(f"\n{paper['summary']}")

                # Join and send message
                message = "\n".join(message_lines)

                # Mattermost has a message limit, split if necessary
                max_message_length = 4000
                if len(message) > max_message_length:
                    # Send digest as first message
                    digest_message = "\n".join(message_lines[:6])
                    self.mattermost_bot.send_message(digest_message)

                    # Send papers in chunks
                    remaining = "\n".join(message_lines[6:])
                    for i in range(0, len(remaining), max_message_length):
                        chunk = remaining[i:i+max_message_length]
                        if chunk.strip():
                            self.mattermost_bot.send_message(chunk)
                else:
                    self.mattermost_bot.send_message(message)

            # Disconnect
            self.mattermost_bot.disconnect()

        except Exception as e:
            print(f"Error sending to Mattermost: {e}")


def main():
    """Main entry point."""
    import argparse

    # Build category choices and help text
    category_choices = list(CATEGORY_CONFIGS.keys())
    category_help = "arXiv category. Available options:\n"
    for cat, config in CATEGORY_CONFIGS.items():
        category_help += f"  {cat}: {config['name']}\n"

    parser = argparse.ArgumentParser(
        description='Generate daily arXiv summaries',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=category_help
    )
    parser.add_argument('--config', help='YAML config file for settings')
    parser.add_argument('--category', default=None, choices=category_choices,
                       help='arXiv category (overrides YAML/env)')
    parser.add_argument('--days', type=int, default=None,
                       help='Days to look back (overrides YAML/env)')
    parser.add_argument('--max-results', type=int, default=None,
                       help='Maximum papers to fetch (overrides YAML/env)')
    parser.add_argument('--min-relevance', type=float, default=None,
                       help='Minimum relevance score 0-1 (overrides YAML/env)')
    parser.add_argument('--preferences', default=None,
                       help='Preferences file (overrides YAML/env)')
    parser.add_argument('--output', help='Output file for digest (overrides YAML/env)')
    parser.add_argument('--format', choices=['text', 'markdown', 'pdf'],
                       default=None,
                       help='Output format (overrides YAML/env)')
    parser.add_argument('--model', default=None,
                       help='Ollama model to use (overrides YAML/env)')
    parser.add_argument('--ollama-url', default=None,
                       help='Ollama API URL (overrides YAML/env)')
    parser.add_argument('--auth-token',
                       help='Bearer token for Ollama authentication (overrides YAML/env)')
    parser.add_argument('--list-categories', action='store_true',
                       help='List all available categories and exit')

    # Email options
    email_group = parser.add_argument_group('email options')
    email_group.add_argument('--email', action='store_true', default=None,
                            help='Send digest via email (overrides YAML/env)')
    email_group.add_argument('--email-to',
                            help='Recipient email address')
    email_group.add_argument('--email-from',
                            help='Sender email address')
    email_group.add_argument('--email-password',
                            help='Sender email password (or set EMAIL_PASSWORD env var)')
    email_group.add_argument('--smtp-server',
                            help='SMTP server address (e.g., smtp.gmail.com)')
    email_group.add_argument('--smtp-port', type=int, default=587,
                            help='SMTP port (default: 587)')

    # Mattermost options
    mattermost_group = parser.add_argument_group('mattermost options')
    mattermost_group.add_argument('--mattermost', action='store_true', default=None,
                                 help='Send digest to Mattermost (overrides YAML/env)')
    mattermost_group.add_argument('--mm-server-url',
                                 help='Mattermost server URL (e.g., https://mattermost.example.com)')
    mattermost_group.add_argument('--mm-bot-token',
                                 help='Mattermost bot access token (or set MATTERMOST_BOT_TOKEN env var)')
    mattermost_group.add_argument('--mm-channel-id',
                                 help='Mattermost channel ID to post messages')
    mattermost_group.add_argument('--mm-content', choices=['summaries', 'pdf'],
                                 default=None,
                                 help='Content to send to Mattermost: text summaries or the generated PDF')

    args = parser.parse_args()

    # Handle --list-categories
    if args.list_categories:
        print("Available arXiv categories:")
        print("=" * 80)
        for cat, config in sorted(CATEGORY_CONFIGS.items()):
            print(f"\n{cat}")
            print(f"  Name: {config['name']}")
            print(f"  Default preferences: {', '.join(config.get('default_preferences', []))}")
        return

    yaml_sections = load_yaml_config(args.config)
    yaml_s = yaml_sections.get('summarizer', {})
    yaml_email = yaml_sections.get('email', {})
    yaml_mm = yaml_sections.get('mattermost', {})

    def first_non_none(*values):
        for v in values:
            if v is not None:
                return v
        return None

    try:
        summarizer_settings = SummarizerSettings(**{
            'preferences_file': first_non_none(args.preferences, yaml_s.get('preferences_file')),
            'category': first_non_none(args.category, yaml_s.get('category')),
            'days_back': first_non_none(args.days, yaml_s.get('days_back')),
            'max_results': first_non_none(args.max_results, yaml_s.get('max_results')),
            'min_relevance': first_non_none(args.min_relevance, yaml_s.get('min_relevance')),
            'output_file': first_non_none(args.output, yaml_s.get('output_file')),
            'output_format': first_non_none(args.format, yaml_s.get('output_format')),
            'model': first_non_none(args.model, yaml_s.get('model')),
            'ollama_url': first_non_none(args.ollama_url, yaml_s.get('ollama_url')),
            'auth_token': first_non_none(args.auth_token, yaml_s.get('auth_token')),
        })

        email_settings = EmailSettings(**{
            'enabled': first_non_none(args.email, yaml_email.get('enabled')),
            'smtp_server': first_non_none(args.smtp_server, yaml_email.get('smtp_server')),
            'smtp_port': first_non_none(args.smtp_port, yaml_email.get('smtp_port')),
            'sender_email': first_non_none(args.email_from, yaml_email.get('sender_email')),
            'sender_password': first_non_none(args.email_password, yaml_email.get('sender_password')),
            'recipient_email': first_non_none(args.email_to, yaml_email.get('recipient_email')),
        })

        mattermost_settings = MattermostSettings(**{
            'enabled': first_non_none(args.mattermost, yaml_mm.get('enabled')),
            'server_url': first_non_none(args.mm_server_url, yaml_mm.get('server_url')),
            'bot_token': first_non_none(args.mm_bot_token, yaml_mm.get('bot_token')),
            'channel_id': first_non_none(args.mm_channel_id, yaml_mm.get('channel_id')),
            'content_mode': first_non_none(args.mm_content, yaml_mm.get('content_mode'), "summaries"),
        })
        print(f"Using summarizer settings: {summarizer_settings}")
        print(f"Using email settings: {email_settings}")
        print(f"Using Mattermost settings: {mattermost_settings}")
    except ValidationError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    summarizer = ArxivSummarizer(
        preferences_file=summarizer_settings.preferences_file,
        ollama_url=summarizer_settings.ollama_url,
        model=summarizer_settings.model,
        auth_token=summarizer_settings.auth_token,
        category=summarizer_settings.category,
        email_config=email_settings if email_settings.is_enabled else None,
        mattermost_config=mattermost_settings if mattermost_settings.is_enabled else None,
    )

    digest = summarizer.run(
        category=summarizer_settings.category,
        days_back=summarizer_settings.days_back,
        max_results=summarizer_settings.max_results,
        min_relevance=summarizer_settings.min_relevance,
        output_file=summarizer_settings.output_file,
        output_format=summarizer_settings.output_format,
        send_email=email_settings.is_enabled,
        send_mattermost=mattermost_settings.is_enabled,
        mattermost_content=mattermost_settings.content_mode,
    )

    if not summarizer_settings.output_file and not email_settings.is_enabled:
        print("\n" + digest)


if __name__ == "__main__":
    main()
