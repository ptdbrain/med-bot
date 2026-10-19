import argparse
import json
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv

        load_dotenv(args.env_file)
    from app.jobs.ingest_jobs import IngestJobQueue
    from app.rag.service import RAGService

    service = RAGService()
    queue = IngestJobQueue(service.settings.metadata_db_path, service.settings.data_dir / "processed" / "job_uploads")
    while True:
        job = queue.run_one(service)
        if job:
            print(json.dumps({"job_id": job["job_id"], "status": job["status"]}), flush=True)
        if args.once:
            return
        if not job:
            time.sleep(1)


if __name__ == "__main__":
    main()
