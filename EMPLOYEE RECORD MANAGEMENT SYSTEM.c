
#include <stdio.h> ///for input output functions like printf, scanf
#include <stdlib.h>  ///memory allocations and process control
#include <conio.h>
#include <windows.h> ///for windows related functions
#include <string.h>  ///string operations

struct emp
{
    char name[40],address[40]; ///name of employee    ///address of employee
    int age; /// age of employee
    int id;  ///id of employee
    float bs; /// basic salary of employee
};

int main()
{

    struct emp e; /// structure variable creation

    char empname[40]; /// string to store name of the employee

    FILE *fp, *ft; /// file pointers
    char another, choice;

    /** open the file in binary read and write mode
    * if the file EMP.DAT already exists then it open that file in read write mode
    * if the file doesn't exit it simply create a new copy
    */
    fp = fopen("EMP.bin","ab+");
    if(fp == NULL)
    {

            printf("Cannot open file");
            exit(1);

    }

    /// sizeo of each record i.e. size of structure variable e


    /// infinite loop continues until the break statement encounter
    while(1)
    {
        system("cls"); ///clear the console window
        printf("\n1. Add Record"); /// option for add record
        printf("\n");
        printf("\n2. List Records"); /// option for showing existing record
        printf("\n");
        printf("\n3. Modify Records"); /// option for editing record
        printf("\n");
        printf("\n4. Delete Records"); /// option for deleting record
        printf("\n");
        printf("\n5. Exit"); /// exit from the program
        printf("\n");
        printf("\nYour Choice: "); /// enter the choice 1, 2, 3, 4, 5
        fflush(stdin); /// flush the input buffer
        printf("\n");
        choice  = getche();
        switch(choice)
        {
        case '1':  /// if user press 1
            system("cls");
              ///back to main screen
            another = 'y';
            while(another == 'y')  /// if user want to add another record
            {
                printf("\nEnter name: ");
                scanf("%s",e.name);
                printf("\nEnter id: ");
                scanf("%d",&e.id);
                printf("\nEnter age: ");
                scanf("%d", &e.age);
                printf("\nEnter basic salary: ");
                scanf("%f", &e.bs);
                printf("\nEnter address: ");
                scanf("%s",e.address);

                fwrite(&e,sizeof(struct emp),1,fp); /// write the record in the file

                printf("\nAdd another record(y/n) ");
                fflush(stdin);
                another = getche();
                rewind(fp);
            }
            break;
        case '2':
            system("cls");
            while(fread(&e,sizeof(struct emp),1,fp)==1)  /// read the file and fetch the record one record per fetch
            {
                printf("\nname = %s   \nid = %d  \nage = %d   \nbasic salary = %.2f   \naddress = %s",e.name,e.id,e.age,e.bs,e.address); /// print the name, age and basic salary
                printf("\n\n");
            }
            getch(); ///wait until pressing the enter
            rewind(fp); ///this moves file cursor to start of the file
            break;

        case '3':  /// if user press 3 then do editing existing record
            system("cls");
            another = 'y';
            while(another == 'y')
            {
                printf("Enter the employee name to modify: ");
                scanf("%s", empname);
                while(fread(&e,sizeof(struct emp),1,fp)==1)  /// fetch all record from file
                {
                    if(strcmp(e.name,empname) == 0)  ///if entered name matches with that in file
                    {
                        printf("\nEnter new name,id , age , bs and address : ");
                        scanf("%s%d%d%f%s",e.name,&e.id,&e.age,&e.bs,e.address);
                        fseek(fp,-sizeof(struct emp),SEEK_CUR); /// move the cursor 1 step back from current position
                        fwrite(&e,sizeof(struct emp),1,fp); /// override the record
                        break;
                    }
                }
                printf("\nModify another record(y/n)");
                fflush(stdin);
                another = getche();
                rewind(fp);
            }
            break;
        case '4':
            system("cls");
            another = 'y';
            while(another == 'y')
            {
                printf("\nEnter name of employee to delete: ");
                scanf("%s",empname);
                ft = fopen("Temp.bin","ab+");  /// create a intermediate file for temporary storage
                rewind(fp); /// move record to starting of file
                while(fread(&e,sizeof(struct emp),1,fp) == 1)  /// read all records from file
                {
                    if(strcmp(e.name,empname) != 0)  /// if the entered record match
                    {
                        fwrite(&e,sizeof(struct emp),1,ft); /// move all records except the one that is to be deleted to temp file
                    }
                }
                fclose(fp);
                fclose(ft);
                remove("EMP.bin"); /// remove the orginal file
                rename("Temp.bin","EMP.bin"); /// rename the temp file to original file name
                fp = fopen("EMP.bin", "ab+");
                printf("Delete another record(y/n)");
                fflush(stdin);
                another = getche();
            }
            break;
        case '5':
            fclose(fp);  /// close the file
            exit(0); /// exit from the program
        }
    }
    return 0;
}

