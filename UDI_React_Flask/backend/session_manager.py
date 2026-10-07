import uuid
SESSIONS={}

def new_session(domain=None):
    sid=str(uuid.uuid4()); SESSIONS[sid]={'domain':domain,'df':None,'profile':None,'messages':[],'last_plan':None,'last_result':None}; return sid

def get(sid):
    return SESSIONS.get(sid)

def reset(sid,domain=None):
    SESSIONS[sid]={'domain':domain,'df':None,'profile':None,'messages':[],'last_plan':None,'last_result':None}
