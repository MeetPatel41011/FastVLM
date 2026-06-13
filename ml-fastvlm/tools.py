import re
import numpy as np

# ─────────────────────────────────────────────
# Tool 1: Scalar Calculator
# ─────────────────────────────────────────────
def execute_calculator(equation: str) -> str:
    """Safe scalar calculator. e.g. "45 * 12", "100 / 4 + 3" """
    math_match = re.search(r'[\d\s\+\-\*\/\.\(\)]+', equation)
    if not math_match:
        return f"Error: No math expression found in '{equation}'"
    safe_equation = math_match.group(0).strip()
    if not safe_equation:
        return f"Error: Empty expression from '{equation}'"
    try:
        import ast, operator
        _op_map = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.USub: operator.neg, ast.UAdd: operator.pos}
        def _eval(node):
            if isinstance(node, ast.Expression): return _eval(node.body)
            elif isinstance(node, ast.Constant): return node.value
            elif isinstance(node, ast.BinOp): return _op_map[type(node.op)](_eval(node.left), _eval(node.right))
            elif isinstance(node, ast.UnaryOp): return _op_map[type(node.op)](_eval(node.operand))
            else: raise ValueError("Unsupported syntax")
        result = _eval(ast.parse(safe_equation, mode='eval'))
        return str(result)
    except Exception as e:
        return f"Error evaluating '{safe_equation}': {e}"


# ─────────────────────────────────────────────
# Tool 2: NumPy Matrix Multiplier
# ─────────────────────────────────────────────
def _parse_complex(s: str) -> complex:
    """Parse complex number strings like 1+i, 4-i, 2, i, 3."""
    s = s.strip().replace(' ', '')
    s = re.sub(r'(?<![0-9])i', '1j', s)
    s = re.sub(r'([0-9])i', r'\1j', s)
    try:
        return complex(s)
    except Exception:
        return complex(0)


def execute_matrix_multiply(problem_text: str) -> str:
    """
    Parses a matrix multiplication problem and computes C = A @ B using NumPy.
    Supports complex numbers using 'i' notation (1+i, 4-i, etc).
    """
    try:
        all_nums = re.findall(
            r'(?<![a-zA-Z])(?:[+\-]?\s*\d+(?:\.\d+)?\s*[+\-]\s*\d*\s*i|[+\-]?\s*\d+(?:\.\d+)?|[+\-]?\s*\d*\s*i)(?![a-zA-Z])',
            problem_text
        )
        filtered = [n for n in all_nums if n.strip() and n.strip() not in ('+', '-')]
        parsed = [_parse_complex(n) for n in filtered]
        if len(parsed) < 4:
            return f"Error: Could not extract enough numbers. Found: {parsed}"
        half = len(parsed) // 2
        a_nums = parsed[:half]
        b_nums = parsed[half:]
        size = int(len(a_nums) ** 0.5)
        if size * size != len(a_nums):
            return f"Error: Matrix is not square (found {len(a_nums)} elements per matrix)"
        A = np.array(a_nums, dtype=complex).reshape(size, size)
        B = np.array(b_nums, dtype=complex).reshape(size, size)
        C = A @ B
        def fmt(v):
            v = complex(v)
            r = round(v.real, 4)
            im = round(v.imag, 4)
            if im == 0:
                return str(int(r) if r == int(r) else r)
            sign = '+' if im >= 0 else '-'
            return f"{int(r) if r == int(r) else r} {sign} {abs(int(im) if im == int(im) else im)}i"
        rows = []
        for row in C:
            rows.append('[' + ',  '.join(fmt(v) for v in row) + ']')
        return 'C =\n  ' + '\n  '.join(rows)
    except Exception as e:
        return f"Error computing matrix multiplication: {e}"


# ─────────────────────────────────────────────
# Tool 3: Web Search (Powered by Tavily AI)
# ─────────────────────────────────────────────
def execute_web_search(query: str) -> str:
    """
    Perform a high-precision web search using Tavily AI.
    Replaces Wikipedia and DuckDuckGo for better accuracy.
    """
    import os
    import requests
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY not found in environment."

    # Clean the query (remove any JSON artifacts if the LLM leaked them)
    if '{' in query:
        # Try to extract content between quotes if it looks like JSON
        match = re.search(r'"(?:text_in_image|query)":\s*"([^"]+)"', query)
        if match:
            query = match.group(1)
        else:
            # Fallback: just remove curly braces and extra quotes
            query = query.replace('{', '').replace('}', '').replace('"', '').strip()
    
    query = query.strip()
    if not query:
        return "Error: Cleaned search query is empty."

    print(f"DEBUG: Sending to Tavily -> '{query}'")
    try:
        # Minimal payload for maximum compatibility across all API key tiers
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": 1,
            "include_answer": True
        }
        response = requests.post(url, json=payload, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            tavily_answer = data.get("answer", "")
            results = data.get("results", [])
            content = results[0].get('content', '') if results else ""
            
            # Prefer the clean Tavily summary if available
            if tavily_answer:
                return f"🌐 Tavily Summary: {tavily_answer}\n\n[Raw Data Context: {content[:200]}...]"
            elif content:
                return f"🌐 Tavily Raw: {content}"
        
        # If Tavily fails or returns nothing, fallback to DuckDuckGo
        return execute_fallback_search(query)
            
    except Exception as e:
        print(f"Tavily Error: {e}")
        return execute_fallback_search(query)

def execute_fallback_search(query: str) -> str:
    """Zero-config fallback using DuckDuckGo."""
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = f"https://html.duckduckgo.com/html/?q={query}"
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            result = soup.find('a', class_='result__a')
            snippet = soup.find('a', class_='result__snippet')
            if result and snippet:
                return f"\U0001f50e Web (Fallback): {snippet.get_text()}"
    except Exception:
        pass
    return f"Sorry, I couldn't find a specific answer for '{query}'."


# ─────────────────────────────────────────────
# Tool Registry — Add new tools here
# ─────────────────────────────────────────────
AVAILABLE_TOOLS = {
    "calculator": execute_calculator,
    "matrix":     execute_matrix_multiply,
    "web_search": execute_web_search,
}