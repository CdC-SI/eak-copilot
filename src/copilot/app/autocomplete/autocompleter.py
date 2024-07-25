import hashlib

from config.base_config import autocomplete_config
from . import matching


class Autocompleter:
    """
    Completer for user input to the copilot
    """

    def __init__(self):
        self.exact_match = matching.ExactMatch()
        self.fuzzy_match = matching.FuzzyMatch(
            threshold=autocomplete_config["fuzzy_match"]["threshold"]
        )
        self.semantic_match = matching.SemanticMatch()

        k = autocomplete_config["results"]["limit"]
        self.limit = 'NULL' if k == matching.INF else k

        self.exact_match_limit = autocomplete_config["exact_match"]["limit"]
        self.fuzzy_match_limit = autocomplete_config["fuzzy_match"]["limit"]
        self.semantic_match_limit = autocomplete_config["semantic_similarity_match"]["limit"]

        self.semantic_matches_cache = {}

    def _cache_key(self, question, language):
        """Generate a cache key based on the question and language."""
        return hashlib.md5(f"{question}_{language}".encode()).hexdigest()

    async def get_autocomplete(self, question: str, language: str = None, k: int = 0):
        """
        Returns matching results according to a defined behaviour

        Parameters
        ----------
        question : str
            question to match
        language : str
            question and results language
        k : int
            number of results to return

        Returns
        -------
        list of str
            a list of matching results
        """
        cache_key = self._cache_key(question, language)
        if cache_key in self.semantic_matches_cache:
            semantic_match = self.semantic_matches_cache[cache_key]
        else:
            semantic_match = await self.semantic_match.match(question, language)
            if self.semantic_match_limit > 0:
                semantic_match = semantic_match[:self.semantic_match_limit]
            self.semantic_matches_cache[cache_key] = semantic_match

        # Get exact and fuzzy matches
        exact_match = await self.exact_match.match(question, language)
        fuzzy_match = await self.fuzzy_match.match(question, language)

        if self.exact_match_limit > 0:
            exact_match = exact_match[:self.exact_match_limit]
        if self.fuzzy_match_limit > 0:
            fuzzy_match = fuzzy_match[:self.fuzzy_match_limit]

        unique_matches = exact_match + fuzzy_match

        # Remove duplicate matches and preserve order
        seen = []
        unique_matches = [seen.append(d) for d in unique_matches if d not in seen]
        unique_matches = seen

        # If the combined results from exact match and fuzzy match are more than 5, return results
        if len(unique_matches) >= 5:
            if self.limit > 0:
                unique_matches = unique_matches[:self.limit]
            return unique_matches

        # If the combined results from exact match and fuzzy match are less than 5, and the question ends with a space or a question mark, perform semantic matching
        if question[-1] == "?":

            # Get semantic matches from cached results
            cache_key = self._cache_key(question, language)
            if cache_key in self.semantic_matches_cache:
                semantic_match = self.semantic_matches_cache[cache_key]
            else:
                semantic_match = await self.semantic_match.match(question, language)
                if self.semantic_match_limit > 0:
                    semantic_match = semantic_match[:self.semantic_match_limit]
                self.semantic_matches_cache[cache_key] = semantic_match

            # Remove duplicates and preserve order
            unique_matches = unique_matches + semantic_match
            unique_matches = [seen.append(d) for d in unique_matches if d not in seen]
            unique_matches = seen

            if self.limit > 0:
                unique_matches = unique_matches[:self.limit]

            return unique_matches

        return unique_matches


autocompleter = Autocompleter()
