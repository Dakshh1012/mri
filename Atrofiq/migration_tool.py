#!/usr/bin/env python3
"""
Migration Script for MRBrain Final Integration
Helps transition from old MRBrain to new MRBrain_final setup
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MRBrainMigrationTool:
    """Tool to help migrate from old MRBrain to MRBrain_final"""
    
    def __init__(self, atrofiq_root: str):
        self.atrofiq_root = Path(atrofiq_root)
        self.old_mrbrain = self.atrofiq_root / "MRBrain"
        self.new_mrbrain = self.atrofiq_root / "MRBrain_final"
        self.backup_dir = self.atrofiq_root / "migration_backup"
        
    def validate_setup(self) -> Dict[str, bool]:
        """Validate the current setup"""
        checks = {
            "atrofiq_root_exists": self.atrofiq_root.exists(),
            "old_mrbrain_exists": self.old_mrbrain.exists(),
            "new_mrbrain_exists": self.new_mrbrain.exists(),
            "docker_compose_exists": (self.atrofiq_root / "docker-compose.yml").exists(),
            "backend_exists": (self.atrofiq_root / "backend").exists(),
        }
        
        logger.info("Validation Results:")
        for check, result in checks.items():
            status = "✓" if result else "✗"
            logger.info(f"  {status} {check}")
        
        return checks
    
    def create_backup(self) -> bool:
        """Create backup of critical files before migration"""
        logger.info("Creating migration backup...")
        
        try:
            if self.backup_dir.exists():
                shutil.rmtree(self.backup_dir)
            
            self.backup_dir.mkdir(exist_ok=True)
            
            # Backup docker-compose.yml
            docker_compose = self.atrofiq_root / "docker-compose.yml"
            if docker_compose.exists():
                shutil.copy2(docker_compose, self.backup_dir / "docker-compose.yml.backup")
                logger.info("✓ Backed up docker-compose.yml")
            
            # Backup backend tasks
            tasks_dir = self.atrofiq_root / "backend" / "app" / "tasks"
            if tasks_dir.exists():
                backup_tasks = self.backup_dir / "tasks_backup"
                backup_tasks.mkdir(exist_ok=True)
                for task_file in tasks_dir.glob("*.py"):
                    shutil.copy2(task_file, backup_tasks / f"{task_file.name}.backup")
                logger.info("✓ Backed up backend tasks")
            
            # Backup celery_app.py
            celery_app = self.atrofiq_root / "backend" / "app" / "celery_app.py"
            if celery_app.exists():
                shutil.copy2(celery_app, self.backup_dir / "celery_app.py.backup")
                logger.info("✓ Backed up celery_app.py")
            
            # Backup main.py
            main_py = self.atrofiq_root / "backend" / "app" / "main.py"
            if main_py.exists():
                shutil.copy2(main_py, self.backup_dir / "main.py.backup")
                logger.info("✓ Backed up main.py")
            
            logger.info(f"✓ Backup completed in: {self.backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Backup failed: {e}")
            return False
    
    def check_model_files(self) -> Dict[str, List[str]]:
        """Check for required model files in MRBrain_final"""
        required_files = {
            "brainage": [
                "BrainAge-Prediction/saved_models/brain_age_pipeline.pkl",
                "BrainAge-Prediction/inference.py",
                "BrainAge-Prediction/brain_age.py",
                "BrainAge-Prediction/brain_age_models.py"
            ],
            "normative": [
                "Normative Modeling/API.py",
                "Normative Modeling/Models",
                "Normative Modeling/Data_Prepared",
            ],
            "config": [
                "requirements.txt",
            ]
        }
        
        results = {category: [] for category in required_files.keys()}
        
        logger.info("Checking MRBrain_final model files...")
        
        for category, files in required_files.items():
            for file_path in files:
                full_path = self.new_mrbrain / file_path
                if full_path.exists():
                    results[category].append(f"✓ {file_path}")
                else:
                    results[category].append(f"✗ {file_path}")
        
        for category, files in results.items():
            logger.info(f"{category.upper()} files:")
            for file_status in files:
                logger.info(f"  {file_status}")
        
        return results
    
    def generate_migration_report(self) -> Dict:
        """Generate comprehensive migration report"""
        logger.info("Generating migration report...")
        
        validation = self.validate_setup()
        model_check = self.check_model_files()
        
        # Check environment configuration
        env_file = self.atrofiq_root / ".env.mrbrain"
        config_file = self.new_mrbrain / "config.py"
        main_api = self.new_mrbrain / "main_api.py"
        dockerfile = self.new_mrbrain / "Dockerfile"
        
        report = {
            "timestamp": "migration_report_generated",
            "validation": validation,
            "model_files": model_check,
            "configuration": {
                "env_file_exists": env_file.exists(),
                "config_py_exists": config_file.exists(),
                "main_api_exists": main_api.exists(),
                "dockerfile_exists": dockerfile.exists(),
            },
            "migration_status": "ready" if all(validation.values()) else "pending",
            "recommendations": []
        }
        
        # Generate recommendations
        if not validation["new_mrbrain_exists"]:
            report["recommendations"].append("❗ MRBrain_final directory not found")
        
        if not report["configuration"]["config_py_exists"]:
            report["recommendations"].append("❗ config.py missing in MRBrain_final")
        
        if not report["configuration"]["main_api_exists"]:
            report["recommendations"].append("❗ main_api.py missing in MRBrain_final")
        
        # Check for missing model files
        for category, files in model_check.items():
            missing = [f for f in files if f.startswith("✗")]
            if missing:
                report["recommendations"].append(f"⚠ Missing {category} files: {len(missing)}")
        
        if not report["recommendations"]:
            report["recommendations"].append("✅ All checks passed - ready for migration")
        
        return report
    
    def save_migration_report(self, report: Dict) -> str:
        """Save migration report to file"""
        report_file = self.atrofiq_root / "migration_report.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Migration report saved to: {report_file}")
        return str(report_file)
    
    def check_docker_setup(self) -> Dict[str, bool]:
        """Check Docker setup for MRBrain_final"""
        docker_compose = self.atrofiq_root / "docker-compose.yml"
        
        checks = {
            "docker_compose_exists": docker_compose.exists(),
            "mrbrain_final_service": False,
            "volume_mounts": False,
            "environment_vars": False
        }
        
        if docker_compose.exists():
            content = docker_compose.read_text()
            checks["mrbrain_final_service"] = "MRBrain_final" in content
            checks["volume_mounts"] = "./MRBrain_final:/app" in content
            checks["environment_vars"] = "MRBRAIN_FINAL_PATH" in content
        
        logger.info("Docker Setup Check:")
        for check, result in checks.items():
            status = "✓" if result else "✗"
            logger.info(f"  {status} {check}")
        
        return checks
    
    def run_complete_migration_check(self) -> bool:
        """Run complete migration check and generate report"""
        logger.info("Running Complete Migration Check")
        logger.info("=" * 50)
        
        # Validate setup
        validation = self.validate_setup()
        if not validation["atrofiq_root_exists"]:
            logger.error("AtrofIQ root directory not found!")
            return False
        
        # Create backup
        if not self.create_backup():
            logger.warning("Backup creation failed, but continuing...")
        
        # Check Docker setup
        self.check_docker_setup()
        
        # Generate and save report
        report = self.generate_migration_report()
        report_file = self.save_migration_report(report)
        
        logger.info("\n" + "=" * 50)
        logger.info("Migration Check Summary:")
        logger.info("=" * 50)
        
        for recommendation in report["recommendations"]:
            logger.info(recommendation)
        
        logger.info(f"\nDetailed report saved to: {report_file}")
        
        migration_ready = report["migration_status"] == "ready"
        logger.info(f"Migration Status: {'✅ READY' if migration_ready else '⏳ PENDING'}")
        
        return migration_ready

def main():
    """Main migration check function"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python migration_tool.py <atrofiq_root_path>")
        print("Example: python migration_tool.py /path/to/Atrofiq")
        sys.exit(1)
    
    atrofiq_root = sys.argv[1]
    
    if not os.path.exists(atrofiq_root):
        print(f"Error: AtrofIQ root directory not found: {atrofiq_root}")
        sys.exit(1)
    
    migration_tool = MRBrainMigrationTool(atrofiq_root)
    success = migration_tool.run_complete_migration_check()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()