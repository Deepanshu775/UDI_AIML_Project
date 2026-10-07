from gemini_client import GeminiService

_service = None

def _get_service():
    global _service
    if _service is None:
        _service = GeminiService()
    return _service

def create_plan(question,profile,domain,history):
    return _get_service().plan(question,profile,domain,history)

def explain(question,plan,result,domain):
    return _get_service().explain(question,plan,result,domain)
