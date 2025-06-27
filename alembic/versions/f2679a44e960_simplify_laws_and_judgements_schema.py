"""simplify_laws_and_judgements_schema

Revision ID: f2679a44e960
Revises: fdc1e8690e8d
Create Date: 2025-06-26 17:05:08.668949

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f2679a44e960'
down_revision: Union[str, None] = 'fdc1e8690e8d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - Simplify laws and judgements tables."""
    # Get connection to check if columns/indexes exist
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    
    # Check laws table
    laws_columns = [col['name'] for col in inspector.get_columns('laws')]
    laws_indexes = [idx['name'] for idx in inspector.get_indexes('laws')]
    
    # ### Remove unnecessary columns from laws table ###
    if 'ix_laws_title' in laws_indexes:
        op.drop_index('ix_laws_title', table_name='laws')
    if 'title' in laws_columns:
        op.drop_column('laws', 'title')
    if 'created_at' in laws_columns:
        op.drop_column('laws', 'created_at')
    if 'updated_at' in laws_columns:
        op.drop_column('laws', 'updated_at')
    if 'metadata_json' in laws_columns:
        op.drop_column('laws', 'metadata_json')
    
    # Check judgements table
    judgements_columns = [col['name'] for col in inspector.get_columns('judgements')]
    judgements_indexes = [idx['name'] for idx in inspector.get_indexes('judgements')]
    
    # ### Remove unnecessary columns from judgements table ###
    if 'ix_judgements_title' in judgements_indexes:
        op.drop_index('ix_judgements_title', table_name='judgements')
    if 'ix_judgements_case_number' in judgements_indexes:
        op.drop_index('ix_judgements_case_number', table_name='judgements')
    if 'title' in judgements_columns:
        op.drop_column('judgements', 'title')
    if 'case_number' in judgements_columns:
        op.drop_column('judgements', 'case_number')
    if 'court' in judgements_columns:
        op.drop_column('judgements', 'court')
    if 'date_of_judgment' in judgements_columns:
        op.drop_column('judgements', 'date_of_judgment')
    if 'created_at' in judgements_columns:
        op.drop_column('judgements', 'created_at')
    if 'updated_at' in judgements_columns:
        op.drop_column('judgements', 'updated_at')
    if 'metadata_json' in judgements_columns:
        op.drop_column('judgements', 'metadata_json')
    
    # Add index for summary in laws table if it doesn't exist
    if 'ix_laws_summary' not in laws_indexes:
        op.create_index('ix_laws_summary', 'laws', ['summary'])


def downgrade() -> None:
    """Downgrade schema - Restore original complex schema."""
    # ### Restore columns to laws table ###
    op.add_column('laws', sa.Column('title', sa.String(), nullable=True))
    op.add_column('laws', sa.Column('created_at', sa.DateTime(), nullable=True))
    op.add_column('laws', sa.Column('updated_at', sa.DateTime(), nullable=True))
    op.add_column('laws', sa.Column('metadata_json', sa.Text(), nullable=True))
    op.create_index('ix_laws_title', 'laws', ['title'])
    
    # ### Restore columns to judgements table ###
    op.add_column('judgements', sa.Column('title', sa.String(), nullable=True))
    op.add_column('judgements', sa.Column('case_number', sa.String(), nullable=True))
    op.add_column('judgements', sa.Column('court', sa.String(), nullable=True))
    op.add_column('judgements', sa.Column('date_of_judgment', sa.DateTime(), nullable=True))
    op.add_column('judgements', sa.Column('created_at', sa.DateTime(), nullable=True))
    op.add_column('judgements', sa.Column('updated_at', sa.DateTime(), nullable=True))
    op.add_column('judgements', sa.Column('metadata_json', sa.Text(), nullable=True))
    op.create_index('ix_judgements_title', 'judgements', ['title'])
    op.create_index('ix_judgements_case_number', 'judgements', ['case_number'])
    
    # Drop summary index
    op.drop_index('ix_laws_summary', table_name='laws')
