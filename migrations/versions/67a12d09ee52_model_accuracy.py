"""model_accuracy

Revision ID: 67a12d09ee52
Revises: d574c9edfe20
Create Date: 2020-09-17 17:06:20.684084

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '67a12d09ee52'
down_revision = 'd574c9edfe20'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('pava_model_accuracy',
    sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
    sa.Column('model_id', postgresql.UUID(as_uuid=True), nullable=True),
    sa.Column('accuracy', sa.Float(), nullable=True),
    sa.Column('num_test_templates', sa.Integer(), nullable=True),
    sa.Column('date_created', postgresql.TIMESTAMP(), nullable=True),
    sa.ForeignKeyConstraint(['model_id'], ['pava_model.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('pava_model_accuracy')
    # ### end Alembic commands ###
