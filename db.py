from sqlalchemy import create_engine
import os
from dotenv import load_dotenv


load_dotenv()



def get_mysql_engine():
    # Assuming you have set up your connection parameters as environment variables
    db_uri = f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
    engine = create_engine(db_uri)
    return engine