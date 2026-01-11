# ArXiv Daily Summary Generator

**Automatically fetch, filter, and summarize arXiv papers based on your research interests.**

---

## Overview

This Python script fetches the latest papers from [arXiv](https://arXiv.org) in your chosen category, filters them based on your research preferences, and generates concise summaries using a local or remote [Ollama](https://ollama.ai) model. It can output the digest in text, Markdown, or PDF format, and optionally send it via email or post it to a Mattermost channel.

---

## Features

- **Customizable Categories**: Choose from a variety of arXiv categories (e.g., astrophysics, machine learning, quantum physics).
- **Relevance Filtering**: Papers are scored and filtered based on your specified research interests.
- **AI-Powered Summaries**: Uses Ollama to generate concise, technical summaries for each paper.
- **Daily Digest**: Compiles a cohesive overview of the most important developments and trends.
- **Multiple Output Formats**: Text, Markdown, or PDF.
- **Email & Mattermost Integration**: Automatically send your digest via email or post it to a Mattermost channel.

---

## Requirements

- Python 3.11+
- Required libraries:
  ```bash
  pip install arxiv requests markdown weasyprint
  ```
- Better, use python poetry for installation
- Optional:
  - For Mattermost integration: `pip install mattermost-api-reference-client`
  - For PDF generation: Install `weasyprint` system dependencies (see [WeasyPrint docs](https://weasyprint.org/)).

---

## Installation

1. Clone this repository or download the script.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have a running Ollama instance (local or remote) with your preferred model (e.g., `llama3.2`).

---

## Configuration

### 1. Preferences File

Create a `preferences.txt` file in the same directory as the script. Add one research interest per line, e.g.:
```
dark energy
cosmic microwave background
large scale structure
```

If the file doesn’t exist, the script will create it with default preferences for your chosen category.

### 2. YAML Configuration (Optional)

You can pass a single YAML file with `--config my_settings.yaml`. CLI flags override YAML, which overrides environment variables and built-in defaults (precedence: CLI > YAML > env > default).

Example YAML:

```yaml
summarizer:
  category: cs.LG
  days_back: 2
  output_format: pdf
  output_file: digest.pdf
  model: llama3.2
  ollama_url: http://localhost:11434
email:
  enabled: false
mattermost:
  enabled: true
  server_url: https://mattermost.example.com
  channel_id: YOUR_CHANNEL_ID
  bot_token: ${MATTERMOST_BOT_TOKEN}
  content_mode: pdf
```

### 3. Environment Variables (Optional)

Pydantic settings also read from `config.env` or the process environment. Prefixes are `ARXIV_`, `EMAIL_`, and `MATTERMOST_`.

- Summarizer: `ARXIV_PREFERENCES_FILE`, `ARXIV_CATEGORY`, `ARXIV_DAYS_BACK`, `ARXIV_MAX_RESULTS`, `ARXIV_MIN_RELEVANCE`, `ARXIV_OUTPUT_FILE`, `ARXIV_OUTPUT_FORMAT`, `ARXIV_MODEL`, `ARXIV_OLLAMA_URL`, `OLLAMA_AUTH_TOKEN`.
- Email: `EMAIL_ENABLED`, `EMAIL_SMTP_SERVER`, `EMAIL_SMTP_PORT`, `EMAIL_SENDER_EMAIL`, `EMAIL_PASSWORD`, `EMAIL_RECIPIENT_EMAIL`.
- Mattermost: `MATTERMOST_ENABLED`, `MATTERMOST_SERVER_URL`, `MATTERMOST_CHANNEL_ID`, `MATTERMOST_BOT_TOKEN`, `MATTERMOST_CONTENT_MODE`.

### 4. Email (Optional)

To send the digest via email, provide the following arguments:
- `--email`: Enable email sending.
- `--email-to`: Recipient email address.
- `--email-from`: Sender email address.
- `--email-password`: Sender email password (or set the `EMAIL_PASSWORD` environment variable).
- `--smtp-server`: SMTP server address (e.g., `smtp.gmail.com`).
- `--smtp-port`: SMTP port (default: `587`).

### 5. Mattermost (Optional)

To post the digest to Mattermost:
- `--mattermost`: Enable Mattermost posting.
- `--mm-server-url`: Mattermost server URL (e.g., `https://mattermost.example.com`).
- `--mm-bot-token`: Bot access token (or set the `MATTERMOST_BOT_TOKEN` environment variable).
- `--mm-channel-id`: Channel ID to post messages.
- `--mm-content`: Choose `summaries` (default) or `pdf` to upload the generated PDF.

---

## Usage

### Basic Usage

Run the script with default settings:
```bash
python arxiv_daily_summary.py
```

### Customize Category and Output

Generate a digest for the `cs.LG` (Machine Learning) category and save it as a PDF:
```bash
python arxiv_daily_summary.py --category cs.LG --format pdf --output digest.pdf
```

### List Available Categories

To see all available categories and their default preferences:
```bash
python arxiv_daily_summary.py --list-categories
```

### Send via Email

```bash
python arxiv_daily_summary.py --email --email-to your@email.com --email-from your@email.com --email-password yourpassword --smtp-server smtp.example.com
```

### Post to Mattermost

```bash
python arxiv_daily_summary.py --mattermost --mm-server-url https://mattermost.example.com --mm-bot-token your_bot_token --mm-channel-id your_channel_id
```

---

## Command-Line Arguments

| Argument               | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `--category`           | arXiv category (default: `astro-ph.CO`).                                   |
| `--days`               | Days to look back (default: `1`).                                          |
| `--max-results`        | Maximum papers to fetch (default: `100`).                                  |
| `--min-relevance`      | Minimum relevance score (0-1, default: `0.1`).                            |
| `--preferences`        | Preferences file (default: `preferences.txt`).                            |
| `--output`             | Output file for digest.                                                    |
| `--format`             | Output format: `text`, `markdown`, or `pdf` (default: `text`).             |
| `--model`              | Ollama model to use (default: `llama3.2`).                                 |
| `--ollama-url`         | Ollama API URL (default: `http://localhost:11434`).                        |
| `--auth-token`         | Bearer token for Ollama authentication (or set `OLLAMA_AUTH_TOKEN` env var).|
| `--config`             | YAML config file path; CLI overrides YAML.                                 |
| `--list-categories`    | List all available categories and exit.                                    |
| `--email`              | Send digest via email.                                                     |
| `--email-to`           | Recipient email address.                                                   |
| `--email-from`         | Sender email address.                                                      |
| `--email-password`     | Sender email password (or set `EMAIL_PASSWORD` env var).                   |
| `--smtp-server`        | SMTP server address.                                                       |
| `--smtp-port`          | SMTP port (default: `587`).                                                |
| `--mattermost`         | Send digest to Mattermost.                                                  |
| `--mm-server-url`      | Mattermost server URL.                                                     |
| `--mm-bot-token`       | Mattermost bot access token (or set `MATTERMOST_BOT_TOKEN` env var).      |
| `--mm-channel-id`      | Mattermost channel ID to post messages.                                    |
| `--mm-content`         | Mattermost content mode: `summaries` or `pdf`.                             |

---

## Example Output

The script generates a digest like this:

```
================================================================================
ArXiv Daily Digest - 2026-01-11
Category: Cosmology and Nongalactic Astrophysics
================================================================================

Your Research Interests: dark energy, cosmic microwave background, large scale structure

--------------------------------------------------------------------------------
DAILY OVERVIEW
--------------------------------------------------------------------------------
Today's papers in cosmology focus on new simulations of dark energy models and improved measurements of the cosmic microwave background. Two papers stand out for their novel approaches to modeling large-scale structure formation...

--------------------------------------------------------------------------------
INDIVIDUAL PAPER SUMMARIES
--------------------------------------------------------------------------------

1. Title: New Constraints on Dark Energy from CMB Lensing
   Authors: Smith, Jones, et al.
   arXiv ID: 2601.00001
   Published: 2026-01-10
   Relevance Score: 0.95
   PDF: https://arxiv.org/pdf/2601.00001

   Summary: This paper presents new constraints on dark energy using CMB lensing data...
```

---

## Troubleshooting

- **Ollama Connection Issues**: Ensure your Ollama server is running and accessible at the specified URL.
- **Email Errors**: Double-check your SMTP settings and credentials.
- **Mattermost Errors**: Verify your bot token and channel ID.
- **PDF Generation**: Install system dependencies for WeasyPrint if you encounter errors.

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

## Contributing

Pull requests and issues are welcome! Feel free to contribute improvements or report bugs.
