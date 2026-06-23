# AI Product Assistant

> A modern AI-powered Amazon product assistant that evolves from a simple Q&A system into a sophisticated, agent-driven application.

![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)

## Getting Started

### Prerequisites

- OpenAI API key
- Google API key
- Groq API key

### Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/thepembeweb/ai-product-assistant.git
   cd ai-product-assistant
   ```

2. **Set up environment variables:**

   ```sh
   cd packages/server
   cp .env.example .env
   ```

   Edit the `.env` file:

   ```env
   OPENAI_API_KEY=your_google_api_key
   GOOGLE_API_KEY=your_google_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

3. **Run the project:**

   ```sh
   make run-docker-compose
   ```

Streamlit application: http://localhost:8501

FastAPI documentation: http://localhost:8000/docs

## This repository uses data provided by the authors of the following paper.

If you use this work, please cite:

```
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```
