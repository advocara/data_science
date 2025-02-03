


import os

from openai import OpenAI


def load_openai_key() -> str:
    """Load OpenAI API key from ~/openai.key file"""
    try:
        with open(os.path.expanduser('~/openai.key'), 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError("Please create ~/openai.key file with your OpenAI API key")
    except Exception as e:
        raise Exception(f"Error loading OpenAI API key: {e}")

def get_openai_client() -> OpenAI:
    """Get OpenAI client with API key"""
    OpenAI.api_key = load_openai_key()
    return OpenAI(api_key=load_openai_key())