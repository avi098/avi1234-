import javax.swing.*;
import java.sql.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.table.DefaultTableModel;

public class ContactTableExample extends JFrame {
    private JTextField nameField, phoneField, emailField;
    private DefaultTableModel tableModel;
   static private String na,ph;

    public ContactTableExample() {
        initializeUI();
    }
    
    static void getdet(String s1,String s2)
    {
        na=s1;
        ph=s2;
    }

    private void initializeUI() {
        setTitle("Contact Table Example");
        setSize(500, 400);
        setDefaultCloseOperation(EXIT_ON_CLOSE);

        // Create title label
        JLabel titleLabel = new JLabel("RAPPORT");
        titleLabel.setFont(new Font("Arial", Font.BOLD, 16));
        titleLabel.setHorizontalAlignment(SwingConstants.CENTER);

        // Create text fields
       
        // Create table with default table model
        tableModel = new DefaultTableModel();
        tableModel.addColumn("Name");
        tableModel.addColumn("Phone");
        tableModel.addColumn("Email");
        JTable table = new JTable(tableModel);

        // Create button
        JButton addButton = new JButton("Search Match");
        addButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                getdet("Shajith","9489425649");
                try{
                 Class.forName("oracle.jdbc.driver.OracleDriver");  
  
Connection con=DriverManager.getConnection(  
"jdbc:oracle:thin:@localhost:1521:xe","dbms","dbms");  
  
Statement stmt=con.createStatement(); 
               ResultSet zz=stmt.executeQuery("select * from register where name='"+na+"' and ph_no='"+ph+"'");  
               
               zz.next();
               String req=zz.getString(8);
               System.out.println(req);
               
               
               ResultSet js=stmt.executeQuery("select * from joinus "); 
               while(js.next())
               {
                   if(req.equals(js.getString(7))==true)
                   {
                       System.out.println(na);
                       tableModel.addRow(new Object[]{js.getString(1),js.getString(2),js.getString(3)});
                       break;
                   }
                       
               }
               

                // Add contact to table
                //tableModel.addRow(new Object[]{name, phone, email});
                }
                catch (ClassNotFoundException ex) {
                    Logger.getLogger(register.class.getName()).log(Level.SEVERE, null, ex);
                } catch (SQLException ex) {
                    Logger.getLogger(register.class.getName()).log(Level.SEVERE, null, ex);
                }

                // Clear text fields
                
            }
        });
        JButton eButton = new JButton("Exit");
        eButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent s) {
               

                // Add contact to table
            System.exit(0);
                //tableModel.addRow(new Object[]{name, phone, email});

                // Clear text fields
                
            }
        }
        );

        // Create panel and add components
        JPanel panel = new JPanel(new GridLayout(4, 2));
        
        panel.add(addButton);
        panel.add(eButton);

        // Add components to frame
        setLayout(new BorderLayout());
        add(titleLabel, BorderLayout.NORTH);
        add(new JScrollPane(table), BorderLayout.CENTER);
        add(panel, BorderLayout.SOUTH);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                new ContactTableExample().setVisible(true);
            }
        });
    }
}
