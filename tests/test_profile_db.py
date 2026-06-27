import os
from datetime import datetime, timedelta, timezone
from src.memory.profile import ProfileMemoryDB

def run_test():
    print("Initializing Profile Memory DB...")
    db = ProfileMemoryDB()
    
    # 1. Test Profile Facts
    print("\n--- Testing Profile Facts ---")
    fact_text = "User prefers Python over Go."
    fact_id = db.add_fact(fact_text)
    print(f"Added Fact ID {fact_id}: '{fact_text}'")
    
    # Test consolidated facts retrieval (includes IDs)
    facts = db.get_all_facts()
    print(f"Retrieved Facts from DB: {facts}")
    assert len(facts) == 1, "Error: Expected exactly 1 fact!"
    assert facts[0]["id"] == fact_id, "Error: Fact ID mismatch!"
    assert facts[0]["fact"] == fact_text, "Error: Fact text mismatch!"
    
    print("Profile Fact retrieval check (with IDs): PASSED")
    
    # Clean up
    deleted = db.delete_fact(fact_id)
    print(f"Deleted Fact ID {fact_id}: {deleted}")
    
    # 2. Test Reminders Queue
    print("\n--- Testing Reminders Queue ---")
    chat_id = "test_chat_123"
    reminder_msg = "Buy milk"
    
    # Schedule a reminder trigger 1 second in the past (to simulate it being due)
    due_time = datetime.now(timezone.utc) - timedelta(seconds=1)
    reminder_id = db.add_reminder(chat_id, reminder_msg, due_time)
    print(f"Scheduled past reminder ID {reminder_id}")
    
    # Retrieve due reminders
    pending = db.get_pending_reminders()
    print(f"Pending due reminders retrieved: {pending}")
    our_reminder = next((r for r in pending if r["id"] == reminder_id), None)
    assert our_reminder is not None, "Error: Scheduled reminder not found in pending!"
    assert our_reminder["reminder_text"] == reminder_msg, "Error: Text mismatch!"
    print("Pending reminder retrieval check: PASSED")
    
    # Mark as sent
    marked = db.mark_reminder_sent(reminder_id)
    print(f"Marked reminder as sent: {marked}")
    
    # Verify no more pending due reminders
    pending_after = db.get_pending_reminders()
    print(f"Pending due reminders after marking: {pending_after}")
    our_reminder_after = next((r for r in pending_after if r["id"] == reminder_id), None)
    assert our_reminder_after is None, "Error: Reminder was marked sent but still retrieved in pending!"
    print("Sent status update check: PASSED")
    
    print("\nAll SQLite database integration checks: PASSED")

if __name__ == "__main__":
    run_test()
