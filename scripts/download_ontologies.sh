#!/bin/bash

# download_ontologies.sh
# Script to download essential ontologies_old from OBO Foundry for OntoSemantics 2.0

set -e  # Exit on any error

# Configuration
ONTOLOGY_DIR="./data/ontologies"
LOG_FILE="./data/ontology_download.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Ontology definitions (portable approach)
ONTOLOGY_NAMES=("mondo" "hp" "chebi" "go" "cl" "uberon" "doid")
ONTOLOGY_URLS=(
    "http://purl.obolibrary.org/obo/mondo.obo"
    "http://purl.obolibrary.org/obo/hp.obo"
    "http://purl.obolibrary.org/obo/chebi.obo"
    "http://purl.obolibrary.org/obo/go.obo"
    "http://purl.obolibrary.org/obo/cl.obo"
    "http://purl.obolibrary.org/obo/uberon.obo"
    "http://purl.obolibrary.org/obo/doid.obo"
)
ONTOLOGY_DESCRIPTIONS=(
    "Monarch Disease Ontology"
    "Human Phenotype Ontology"
    "Chemical Entities of Biological Interest"
    "Gene Ontology"
    "Cell Ontology"
    "Uber-anatomy Ontology"
    "Human Disease Ontology"
)
ONTOLOGY_SIZES=(
    "~50MB"
    "~15MB"
    "~200MB"
    "~80MB"
    "~5MB"
    "~40MB"
    "~25MB"
)

# Essential ontologies_old for quick setup (indices)
ESSENTIAL_INDICES=(0 1 2)  # mondo, hp, chebi

# Function to get ontology info by name
get_ontology_index() {
    local name="$1"
    for i in "${!ONTOLOGY_NAMES[@]}"; do
        if [[ "${ONTOLOGY_NAMES[$i]}" == "$name" ]]; then
            echo "$i"
            return 0
        fi
    done
    echo "-1"
}

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to show help
show_help() {
    echo "OntoSemantics 2.0 - Ontology Downloader"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -a, --all          Download all ontologies"
    echo "  -e, --essential    Download only essential ontologies (mondo, hp, chebi)"
    echo "  -s, --select       Select specific ontologies interactively"
    echo "  -l, --list         List available ontologies"
    echo "  -f, --force        Force re-download even if files exist"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --essential     # Quick setup for demo"
    echo "  $0 --all           # Download everything"
    echo "  $0 --select        # Choose what to download"
}

# Function to list available ontologies_old
list_ontologies() {
    echo "Available Ontologies:"
    echo "===================="
    for i in "${!ONTOLOGY_NAMES[@]}"; do
        printf "%-10s %-40s %s\n" "${ONTOLOGY_NAMES[$i]}" "${ONTOLOGY_DESCRIPTIONS[$i]}" "${ONTOLOGY_SIZES[$i]}"
    done
}

# Function to check if a file exists and is recent (less than 7 days old)
is_recent() {
    local file="$1"
    if [[ -f "$file" ]]; then
        # Check if file is less than 7 days old
        if [[ $(find "$file" -mtime -7 2>/dev/null) ]]; then
            return 0
        fi
    fi
    return 1
}

# Function to download a single ontology by index
download_ontology_by_index() {
    local index="$1"
    local force_download="$2"

    if [[ $index -lt 0 || $index -ge ${#ONTOLOGY_NAMES[@]} ]]; then
        print_error "Invalid ontology index: $index"
        return 1
    fi

    local onto_name="${ONTOLOGY_NAMES[$index]}"
    local url="${ONTOLOGY_URLS[$index]}"
    local description="${ONTOLOGY_DESCRIPTIONS[$index]}"
    local size="${ONTOLOGY_SIZES[$index]}"
    local filename="${ONTOLOGY_DIR}/${onto_name}.obo"

    # Check if file exists and is recent
    if [[ "$force_download" != "true" ]] && is_recent "$filename"; then
        print_warning "$onto_name.obo is recent (less than 7 days old), skipping download"
        return 0
    fi

    print_status "Downloading $onto_name ($description, $size)..."

    # Download with progress bar and resume capability
    if curl -L --fail --continue-at - --progress-bar -o "$filename" "$url" 2>> "$LOG_FILE"; then
        print_success "Downloaded $onto_name.obo"

        # Verify the file is valid OBO format
        if head -10 "$filename" | grep -q "format-version:" 2>/dev/null; then
            print_success "$onto_name.obo appears to be valid OBO format"
        else
            print_warning "$onto_name.obo may not be valid OBO format"
        fi

        # Show file size
        local file_size=$(du -h "$filename" 2>/dev/null | cut -f1)
        print_status "File size: $file_size"

    else
        print_error "Failed to download $onto_name"
        echo "$(date): Failed to download $onto_name from $url" >> "$LOG_FILE"
        return 1
    fi
}

# Function to download ontology by name
download_ontology() {
    local onto_name="$1"
    local force_download="$2"

    local index=$(get_ontology_index "$onto_name")
    if [[ $index -eq -1 ]]; then
        print_error "Unknown ontology: $onto_name"
        return 1
    fi

    download_ontology_by_index "$index" "$force_download"
}

# Function for interactive selection
interactive_selection() {
    echo "Select ontologies to download:"
    echo "============================="

    for i in "${!ONTOLOGY_NAMES[@]}"; do
        echo "$((i+1))) ${ONTOLOGY_NAMES[$i]} - ${ONTOLOGY_DESCRIPTIONS[$i]} (${ONTOLOGY_SIZES[$i]})"
    done

    echo ""
    echo "Enter numbers separated by spaces (e.g., 1 2 3), or 'all' for everything:"
    read -r selection

    selected_indices=()

    if [[ "$selection" == "all" ]]; then
        for i in "${!ONTOLOGY_NAMES[@]}"; do
            selected_indices+=("$i")
        done
    else
        for num in $selection; do
            if [[ $num =~ ^[0-9]+$ ]] && [[ $num -ge 1 ]] && [[ $num -le ${#ONTOLOGY_NAMES[@]} ]]; then
                selected_indices+=("$((num-1))")
            fi
        done
    fi

    if [[ ${#selected_indices[@]} -eq 0 ]]; then
        print_error "No valid selections made"
        exit 1
    fi

    selected_names=()
    for i in "${selected_indices[@]}"; do
        selected_names+=("${ONTOLOGY_NAMES[$i]}")
    done

    echo "Selected: ${selected_names[*]}"
}

# Function to estimate total download size
estimate_size() {
    local indices=("$@")
    local total_mb=0

    for i in "${indices[@]}"; do
        local size="${ONTOLOGY_SIZES[$i]}"
        # Extract number from size (rough estimation)
        local mb=$(echo "$size" | grep -o '[0-9]\+' | head -1)
        total_mb=$((total_mb + mb))
    done

    echo "Estimated total download: ~${total_mb}MB"
}

# Main execution
main() {
    local download_all=false
    local download_essential=false
    local interactive=false
    local force_download=false

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -a|--all)
                download_all=true
                shift
                ;;
            -e|--essential)
                download_essential=true
                shift
                ;;
            -s|--select)
                interactive=true
                shift
                ;;
            -l|--list)
                list_ontologies
                exit 0
                ;;
            -f|--force)
                force_download=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Check if no options provided
    if [[ "$download_all" == false && "$download_essential" == false && "$interactive" == false ]]; then
        print_warning "No download option specified. Use --help for options."
        echo "Quick start: $0 --essential"
        exit 1
    fi

    # Create ontology directory
    mkdir -p "$ONTOLOGY_DIR"

    # Initialize log file
    echo "$(date): Starting ontology download" > "$LOG_FILE"

    print_status "OntoSemantics 2.0 - Ontology Downloader"
    print_status "Ontology directory: $ONTOLOGY_DIR"
    print_status "Log file: $LOG_FILE"
    echo ""

    # Determine which ontologies_old to download
    local to_download_indices=()

    if [[ "$download_all" == true ]]; then
        for i in "${!ONTOLOGY_NAMES[@]}"; do
            to_download_indices+=("$i")
        done
        print_status "Downloading ALL ontologies"
    elif [[ "$download_essential" == true ]]; then
        to_download_indices=("${ESSENTIAL_INDICES[@]}")
        print_status "Downloading ESSENTIAL ontologies for quick setup"
    elif [[ "$interactive" == true ]]; then
        interactive_selection
        to_download_indices=("${selected_indices[@]}")
    fi

    # Show download plan
    download_names=()
    for i in "${to_download_indices[@]}"; do
        download_names+=("${ONTOLOGY_NAMES[$i]}")
    done
    echo "Ontologies to download: ${download_names[*]}"
    estimate_size "${to_download_indices[@]}"

    if [[ "$force_download" == true ]]; then
        print_warning "Force mode: will re-download existing files"
    fi

    echo ""
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Download cancelled"
        exit 0
    fi

    # Download ontologies_old
    local success_count=0
    local total_count=${#to_download_indices[@]}

    print_status "Starting downloads..."
    echo ""

    for i in "${to_download_indices[@]}"; do
        if download_ontology_by_index "$i" "$force_download"; then
            ((success_count++))
        fi
        echo ""
    done

    # Summary
    echo "==============================="
    print_status "Download Summary"
    echo "==============================="
    print_success "Successfully downloaded: $success_count/$total_count ontologies"

    if [[ $success_count -lt $total_count ]]; then
        print_warning "Some downloads failed. Check $LOG_FILE for details."
    fi

    print_status "Ontologies saved to: $ONTOLOGY_DIR"
    print_status "Ready for OntoSemantics 2.0!"

    # Show next steps
    echo ""
    echo "Next steps:"
    echo "1. Run: python setup_ontologies.py"
    echo "2. Test with: python test_pronto_loading.py"
}

# Check dependencies
if ! command -v curl &> /dev/null; then
    print_error "curl is required but not installed"
    exit 1
fi

# Run main function
main "$@"