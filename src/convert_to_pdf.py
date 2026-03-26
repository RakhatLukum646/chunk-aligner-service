#!/usr/bin/env python3
"""
Script to convert doc, docx, and xlsx files to PDF using a local conversion endpoint.
Supports pre-conversion of .doc and .xlsx files using LibreOffice.
"""

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
import requests
from typing import List, Optional, Tuple


class FileConverter:
    """Handle conversion of office files to PDF via API endpoint."""
    
    def __init__(self, endpoint_url: str = "http://localhost:8000/convert-to-pdf/", 
                 use_libreoffice: bool = True):
        """
        Initialize the converter.
        
        Args:
            endpoint_url: The URL of the conversion endpoint
            use_libreoffice: Whether to use LibreOffice for pre-conversion
        """
        self.endpoint_url = endpoint_url
        self.supported_extensions = {'.doc', '.docx', '.xlsx'}
        self.use_libreoffice = use_libreoffice
        self.temp_dir = None
        
        # Check if LibreOffice is available
        if use_libreoffice:
            self.libreoffice_available = self._check_libreoffice()
            if not self.libreoffice_available:
                print("⚠ Warning: LibreOffice not found. .doc and .xlsx files will fail.")
                print("  Install with: sudo apt-get install libreoffice")
        else:
            self.libreoffice_available = False
    
    def _check_libreoffice(self) -> bool:
        """Check if LibreOffice is available."""
        try:
            result = subprocess.run(
                ['soffice', '--version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            try:
                result = subprocess.run(
                    ['libreoffice', '--version'],
                    capture_output=True,
                    timeout=5
                )
                return result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False
    
    def _get_libreoffice_command(self) -> str:
        """Get the LibreOffice command name."""
        for cmd in ['soffice', 'libreoffice']:
            try:
                result = subprocess.run(
                    [cmd, '--version'],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return cmd
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        return 'soffice'
    
    def _convert_with_libreoffice(self, file_path: Path, output_format: str, 
                                  output_dir: Path) -> Optional[Path]:
        """
        Convert a file using LibreOffice.
        
        Args:
            file_path: Path to the file to convert
            output_format: Target format (docx or pdf)
            output_dir: Directory for output
            
        Returns:
            Path to converted file or None if failed
        """
        try:
            cmd = self._get_libreoffice_command()
            
            # LibreOffice command
            result = subprocess.run(
                [
                    cmd,
                    '--headless',
                    '--convert-to', output_format,
                    '--outdir', str(output_dir),
                    str(file_path)
                ],
                capture_output=True,
                timeout=60,
                text=True
            )
            
            if result.returncode != 0:
                print(f"  LibreOffice error: {result.stderr}")
                return None
            
            # Find the converted file
            expected_name = f"{file_path.stem}.{output_format}"
            output_file = output_dir / expected_name
            
            if output_file.exists():
                return output_file
            
            return None
            
        except subprocess.TimeoutExpired:
            print(f"  LibreOffice conversion timed out")
            return None
        except Exception as e:
            print(f"  LibreOffice conversion error: {e}")
            return None
    
    def find_files(self, directory: str) -> List[Path]:
        """
        Find all supported files in the given directory.
        
        Args:
            directory: Path to the directory to search
            
        Returns:
            List of Path objects for supported files
        """
        directory_path = Path(directory).expanduser().resolve()
        
        if not directory_path.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory}\n"
                f"Resolved path: {directory_path}\n"
                f"Hint: Use absolute path (e.g., /home/user/...) or relative path without leading './' for absolute paths"
            )
        
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")
        
        files = []
        for ext in self.supported_extensions:
            files.extend(directory_path.glob(f"*{ext}"))
            # Also search recursively if needed
            files.extend(directory_path.glob(f"**/*{ext}"))
        
        # Remove duplicates
        files = list(set(files))
        
        return sorted(files)
    
    def convert_file(self, file_path: Path, output_dir: Optional[str] = None) -> Tuple[bool, str]:
        """
        Convert a single file to PDF.
        
        Args:
            file_path: Path to the file to convert
            output_dir: Directory to save the PDF
            
        Returns:
            Tuple of (success, method_used)
        """
        try:
            print(f"Converting: {file_path.name}...")
            
            file_to_convert = file_path
            temp_file = None
            method = "endpoint"
            
            # Handle .doc files - convert to .docx first
            if file_path.suffix.lower() == '.doc' and self.libreoffice_available:
                print(f"  Pre-converting .doc to .docx using LibreOffice...")
                if self.temp_dir is None:
                    self.temp_dir = tempfile.mkdtemp()
                
                temp_docx = self._convert_with_libreoffice(
                    file_path, 'docx', Path(self.temp_dir)
                )
                
                if temp_docx and temp_docx.exists():
                    file_to_convert = temp_docx
                    temp_file = temp_docx
                    method = "libreoffice+endpoint"
                else:
                    print(f"✗ Failed to pre-convert {file_path.name}")
                    return False, "failed"
            
            # Handle .xlsx files - convert directly to PDF with LibreOffice
            elif file_path.suffix.lower() == '.xlsx' and self.libreoffice_available:
                print(f"  Converting .xlsx to PDF using LibreOffice...")
                output_path = Path(output_dir) / f"{file_path.stem}.pdf"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                temp_pdf = self._convert_with_libreoffice(
                    file_path, 'pdf', Path(output_dir)
                )
                
                if temp_pdf and temp_pdf.exists():
                    print(f"✓ Successfully converted: {file_path.name} -> {output_path}")
                    return True, "libreoffice"
                else:
                    print(f"✗ Failed to convert {file_path.name} with LibreOffice")
                    return False, "failed"
            
            # Send to endpoint (for .docx files or pre-converted .doc files)
            with open(file_to_convert, 'rb') as f:
                files = {'file': (file_to_convert.name, f, self._get_mime_type(file_to_convert))}
                
                response = requests.post(self.endpoint_url, files=files)
                
                if response.status_code == 200:
                    output_path = Path(output_dir) / f"{file_path.stem}.pdf"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(output_path, 'wb') as pdf_file:
                        pdf_file.write(response.content)
                    
                    print(f"✓ Successfully converted: {file_path.name} -> {output_path}")
                    
                    # Clean up temp file
                    if temp_file and temp_file.exists():
                        temp_file.unlink()
                    
                    return True, method
                else:
                    print(f"✗ Failed to convert {file_path.name}: HTTP {response.status_code}")
                    print(f"  Response: {response.text}")
                    return False, "failed"
                    
        except requests.exceptions.RequestException as e:
            print(f"✗ Network error converting {file_path.name}: {e}")
            return False, "failed"
        except Exception as e:
            print(f"✗ Error converting {file_path.name}: {e}")
            return False, "failed"
    
    def cleanup(self):
        """Clean up temporary directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _get_mime_type(self, file_path: Path) -> str:
        """Get the MIME type for a file based on its extension."""
        mime_types = {
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        }
        return mime_types.get(file_path.suffix.lower(), 'application/octet-stream')
    
    def convert_directory(self, directory: str, output_dir: str, 
                         recursive: bool = True) -> dict:
        """
        Convert all supported files in a directory.
        
        Args:
            directory: Path to the directory containing files
            output_dir: Directory to save PDFs
            recursive: Whether to search subdirectories
            
        Returns:
            Dictionary with conversion statistics
        """
        # Resolve directory path
        directory_path = Path(directory).expanduser().resolve()
        
        # Create output directory
        output_path = Path(output_dir).expanduser().resolve()
        os.makedirs(output_path, exist_ok=True)
        
        # Find all files
        files = self.find_files(directory)
        
        if not files:
            print(f"No supported files found in {directory}")
            return {'total': 0, 'success': 0, 'failed': 0, 'by_method': {}}
        
        print(f"\nFound {len(files)} file(s) to convert:")
        for file in files:
            try:
                rel_path = file.relative_to(directory_path)
                print(f"  - {rel_path}")
            except ValueError:
                print(f"  - {file.name}")
        print(f"\nOutput directory: {output_path}")
        
        if self.libreoffice_available:
            print("✓ LibreOffice available for .doc and .xlsx conversion\n")
        else:
            print("⚠ LibreOffice not available - .doc and .xlsx files will fail\n")
        
        # Convert each file
        stats = {
            'total': len(files), 
            'success': 0, 
            'failed': 0,
            'by_method': {
                'endpoint': 0,
                'libreoffice': 0,
                'libreoffice+endpoint': 0,
                'failed': 0
            }
        }
        
        for file_path in files:
            success, method = self.convert_file(file_path, output_dir)
            if success:
                stats['success'] += 1
                stats['by_method'][method] += 1
            else:
                stats['failed'] += 1
                stats['by_method']['failed'] += 1
        
        # Clean up temp files
        self.cleanup()
        
        # Print summary
        print("\n" + "="*60)
        print(f"Conversion Summary:")
        print(f"  Total files:      {stats['total']}")
        print(f"  Successfully:     {stats['success']}")
        print(f"  Failed:           {stats['failed']}")
        print(f"\nBy Method:")
        print(f"  Endpoint only:    {stats['by_method']['endpoint']}")
        print(f"  LibreOffice:      {stats['by_method']['libreoffice']}")
        print(f"  LO + Endpoint:    {stats['by_method']['libreoffice+endpoint']}")
        print(f"\n  Output location:  {output_path}")
        print("="*60)
        
        return stats


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Convert doc, docx, and xlsx files to PDF using a local endpoint.'
    )
    parser.add_argument(
        'directory',
        help='Directory containing files to convert (use absolute path like /home/user/... or relative like ./mydir)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output directory for PDF files (default: ./converted_pdfs)',
        default='./converted_pdfs'
    )
    parser.add_argument(
        '-e', '--endpoint',
        help='Conversion endpoint URL',
        default='http://localhost:8000/convert-to-pdf/'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Do not search subdirectories'
    )
    parser.add_argument(
        '--no-libreoffice',
        action='store_true',
        help='Disable LibreOffice pre-conversion (will fail on .doc and .xlsx)'
    )
    
    args = parser.parse_args()
    
    # Create converter
    converter = FileConverter(
        endpoint_url=args.endpoint,
        use_libreoffice=not args.no_libreoffice
    )
    
    # Convert files
    try:
        stats = converter.convert_directory(
            directory=args.directory,
            output_dir=args.output,
            recursive=not args.no_recursive
        )
        
        # Exit with appropriate code
        sys.exit(0 if stats['failed'] == 0 else 1)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
