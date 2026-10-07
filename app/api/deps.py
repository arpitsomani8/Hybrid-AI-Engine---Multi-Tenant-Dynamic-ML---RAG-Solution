from typing import Optional
from fastapi import Header, HTTPException, Request

def get_tenant_id(
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID", description="Tenant Identifier")
) -> str:
    """
    Enforces and resolves tenant context from X-Tenant-ID request header.
    """
    if not x_tenant_id or not x_tenant_id.strip():
        raise HTTPException(
            status_code=400,
            detail="Missing required header 'X-Tenant-ID'. Multi-tenant isolation requires a valid tenant ID."
        )
    return x_tenant_id.strip()
