from pydantic import BaseModel, Field, model_validator
from typing import Optional

class SubnetConfig(BaseModel):
    embed_dim: Optional[int] = Field(None, gt=0, description="Embedding dimension, must be a multiple of 2")
    num_heads: Optional[int] = Field(None, gt=0, description="Number of heads, must be a multiple of 2")
    mlp_dim: Optional[int] = Field(None, gt=0)
    num_layers: Optional[int] = Field(None, gt=0)

    @model_validator(mode='after')
    def validate_multiples_of_two(self) -> 'SubnetConfig':
        if self.embed_dim is not None and self.embed_dim % 2 != 0:
            raise ValueError("embed_dim must be a multiple of 2")
        if self.num_heads is not None and self.num_heads % 2 != 0:
            raise ValueError("num_heads must be a multiple of 2")
        
        # Additionally, if both are provided, embed_dim must be divisible by num_heads
        if self.embed_dim is not None and self.num_heads is not None:
            if self.embed_dim % self.num_heads != 0:
                raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})")
                
        return self
