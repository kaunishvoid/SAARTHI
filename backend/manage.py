"""Administrative commands for provisioning officer accounts without public self-enrollment."""

import argparse
from getpass import getpass

from werkzeug.security import generate_password_hash

from .app import create_app
from .app.extensions import db
from .app.models import OfficerProfile, User


def create_officer(args) -> int:
    app = create_app()
    with app.app_context():
        email = args.email.strip().lower()
        if User.query.filter_by(email=email).first():
            print("An account with that email already exists.")
            return 1
        password = getpass("Officer password (10+ characters): ")
        confirmation = getpass("Confirm password: ")
        if len(password) < 10 or password != confirmation:
            print("Passwords must match and contain at least 10 characters.")
            return 1
        user = User(full_name=args.name.strip(), email=email, password_hash=generate_password_hash(password, method="scrypt"), role="officer", is_demo=args.demo)
        db.session.add(user)
        db.session.flush()
        db.session.add(OfficerProfile(user_id=user.id, display_name=user.full_name, organization=args.organization, is_demo=args.demo))
        db.session.commit()
        print(f"Officer account created for {email} (demo={args.demo}). No availability was seeded.")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    officer = commands.add_parser("create-officer", help="provision a named officer account")
    officer.add_argument("--email", required=True)
    officer.add_argument("--name", required=True)
    officer.add_argument("--organization")
    officer.add_argument("--demo", action="store_true", help="mark this explicitly as a demo account")
    officer.set_defaults(handler=create_officer)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
