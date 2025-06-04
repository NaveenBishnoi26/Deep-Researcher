# Deep Researcher - Advanced Research Assistant

Deep Researcher is an intelligent research assistant that helps users conduct comprehensive research on any topic. It uses advanced AI models to analyze, synthesize, and present research findings in a structured and professional format.

## Features

- 🤖 **Intelligent Query Understanding**: Automatically clarifies ambiguous research queries through interactive questioning
- 📚 **Comprehensive Research**: Gathers information from multiple sources including academic papers, web articles, and databases
- 📊 **Structured Reports**: Generates well-organized research reports with proper sections and citations
- 🔍 **Smart Filtering**: Uses relevance scoring to ensure high-quality, focused research results
- 📝 **Multiple Export Formats**: Supports PDF, DOCX, and HTML export options
- 🎯 **Customizable Parameters**: Adjust relevance thresholds and model settings to suit your needs

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd final_ver
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```bash
streamlit run frontend.py
```

2. Access the web interface at `http://localhost:8501`

3. Enter your research query in the text area

4. If the query needs clarification, answer the follow-up questions

5. Wait for the research process to complete

6. View and download your research report in your preferred format

## Project Structure

```
final_ver/
├── backend/               # Core research pipeline components
│   ├── agents.py         # AI agent implementations
│   ├── clarify.py        # Query clarification logic
│   ├── gather.py         # Data gathering functionality
│   ├── knowledge_base.py # Knowledge base management
│   └── ...
├── frontend.py           # Streamlit web interface
├── app.py               # Main application logic
├── requirements.txt     # Project dependencies
└── run.sh              # Startup script
```

## Key Components

### Query Clarification
- Interactive questioning to understand research requirements
- Context-aware query enhancement
- Multiple question handling

### Research Pipeline
- Intelligent data gathering from multiple sources
- Relevance-based filtering
- Structured report generation

### Export System
- PDF export with proper formatting
- DOCX export with styles
- HTML export for web viewing

## Configuration

The application can be configured through:

1. Environment Variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - Additional API keys for various services

2. UI Settings:
   - Model selection (gpt-4o-mini, gpt-4, gpt-3.5-turbo)
   - Temperature control
   - Relevance threshold
   - Output format selection

## Dependencies

Key dependencies include:
- Streamlit for the web interface
- LangChain for AI pipeline
- OpenAI for language models
- TensorFlow for ML components
- Various document processing libraries

See `requirements.txt` for the complete list.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for providing the language models
- Streamlit for the web framework
- All contributors and users of the project

## Support

For support, please:
1. Check the documentation
2. Open an issue in the repository
3. Contact the development team

## Roadmap

- [ ] Enhanced multi-modal research capabilities
- [ ] Real-time collaboration features
- [ ] Advanced citation management
- [ ] Custom research templates
- [ ] API access for integration 