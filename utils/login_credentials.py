"""
This module provides an login class that bundles all authentification methods

Requirements:
- Dotenv
"""

import os
from dotenv import load_dotenv, find_dotenv


class LoginCredentials:
    """
    This class bundles all authentification keys for improved security
    """

    def __init__(self) -> None:
        load_dotenv(find_dotenv("./tokens.env"))
        self.wandb_key = os.getenv("YOUR_WANDB_KEY")
