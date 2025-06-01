from sorter import HazelnutSorter

def main():
    model_path = r'best.pt'
    
    try:
        sorter = HazelnutSorter(model_path)
        sorter.run()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
