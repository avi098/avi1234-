import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.sql.*;
import java.util.logging.Level;
import java.util.logging.Logger;
//import java.time.LocalDate;

public class Project extends JFrame {
    private JPanel mainPanel;
    private JTextField phNoTextField;
    private JTextField nameTextField;
    private JTextField mailTextField;
    private JTextField addressTextField;
    private JTextField typeTextField;
    private JTextField reqlabel;
    private JComboBox<String> reqLabelComboBox;
    private JComboBox<String> genComboBox;
   // private final JLabel reqLabel;
    private JButton registerButton;
            

    public Project() {
        setTitle("Join Us Page");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(800, 600);
        setLocationRelativeTo(null);
        setLayout(new BorderLayout());

        mainPanel = new JPanel();
        mainPanel.setLayout(new GridBagLayout());
        GridBagConstraints constraints = new GridBagConstraints();
        constraints.insets = new Insets(40, 10, 10, 10);

        JLabel headingLabel = new JLabel("Join Us");
        headingLabel.setFont(new Font("TimesNewRoman", Font.BOLD, 25));
        constraints.gridx = 0;
        constraints.gridy = 0;
        constraints.gridwidth = 2;
        constraints.anchor = GridBagConstraints.CENTER;
        mainPanel.add(headingLabel, constraints);

        JLabel phNoLabel = new JLabel("Name:");
        phNoLabel.setFont(new Font("TimesNewRoman", Font.BOLD, 13));
        constraints.gridx = 0;
        constraints.gridy = 1;
        constraints.gridwidth = 1;
        mainPanel.add(phNoLabel, constraints);

        phNoTextField = new JTextField();
        phNoTextField.setPreferredSize(new Dimension(200, 25));
        constraints.gridx = 1;
        mainPanel.add(phNoTextField, constraints);

        JLabel nameLabel = new JLabel("Phone number:");
        nameLabel.setFont(new Font("TimesNewRoman", Font.BOLD, 13));
        constraints.gridx = 0;
        constraints.gridy = 2;
        mainPanel.add(nameLabel, constraints);

        nameTextField = new JTextField(30);
        nameTextField.setPreferredSize(new Dimension(200, 25));
        constraints.gridx = 1;
        mainPanel.add(nameTextField, constraints);
        
        JLabel mailLabel = new JLabel("Emai;:");
        nameLabel.setFont(new Font("TimesNewRoman", Font.BOLD, 13));
        constraints.gridx = 0;
        constraints.gridy = 3;
        mainPanel.add(mailLabel, constraints);

        mailTextField = new JTextField(30);
        nameTextField.setPreferredSize(new Dimension(200, 25));
        constraints.gridx = 1;
        mainPanel.add(mailTextField, constraints);

        JLabel addressLabel = new JLabel("Address:");
        addressLabel.setFont(new Font("TimesNewRoman", Font.BOLD, 13));
        constraints.gridx = 0;
        constraints.gridy = 4;
        mainPanel.add(addressLabel, constraints);

        addressTextField = new JTextField(30);
        addressTextField.setPreferredSize(new Dimension(200, 25));
        constraints.gridx = 1;
        mainPanel.add(addressTextField, constraints);

        JLabel genderLabel = new JLabel("Gender:");
        genderLabel.setFont(new Font("TimesNewRoman", Font.BOLD, 13));
        constraints.gridx = 0;
        constraints.gridy = 5;
        mainPanel.add(genderLabel, constraints);

        String[] gen = {"Male", "Female", "Others"};
        genComboBox = new JComboBox<>(gen);
        constraints.gridx = 1;
        mainPanel.add(genComboBox, constraints);

        JLabel typeLabel = new JLabel("Available day to help:");
        typeLabel.setFont(new Font("TimesNewRoman", Font.BOLD, 13));
        constraints.gridx = 0;
        constraints.gridy = 6;
        mainPanel.add(typeLabel, constraints);

        typeTextField = new JTextField();
        typeTextField.setPreferredSize(new Dimension(200, 25));
        constraints.gridx = 1;
        mainPanel.add(typeTextField, constraints);
        
        JLabel reqLabel = new JLabel("Requirement:");
        reqLabel.setFont(new Font("TimesNewRoman", Font.BOLD, 13));
        constraints.gridx = 0;
        constraints.gridy = 7;
        mainPanel.add(reqLabel, constraints);

        String[] req = {"Wheel chair", "Hearing Aid", "Walking Cane", "Prosthetic Device", "Employment", "Medicine", "Outfits"};
        reqLabelComboBox = new JComboBox<>(req);
        constraints.gridx = 1;
        mainPanel.add(reqLabelComboBox, constraints);

        registerButton = new JButton("Join");
        constraints.gridx = 0;
        constraints.gridy = 8;
        constraints.gridwidth = 2;
        constraints.anchor = GridBagConstraints.CENTER;
        mainPanel.add(registerButton, constraints);

        getContentPane().add(mainPanel, BorderLayout.CENTER);
        addListeners();
    }


    private void addComponents() {
        // Set the layout manager for the content pane
        getContentPane().setLayout(new BorderLayout());

        // Add spacing around the main panel
        mainPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        // Add the main panel to the content pane
        getContentPane().add(mainPanel, BorderLayout.CENTER);
    }
       private void addListeners() {
            // Save data to the database
             registerButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                // Handle create account button action
                // Add your logic here
   try{
            // Load the Oracle JDBC driver class
            Class.forName("oracle.jdbc.driver.OracleDriver");  
  
Connection con=DriverManager.getConnection(  
"jdbc:oracle:thin:@localhost:1521:xe","dbms","dbms");  
  
Statement stmt=con.createStatement(); 
            
             
            String name = phNoTextField.getText();
            String phno = nameTextField.getText();
            String email = mailTextField.getText();
            String address = addressTextField.getText();
            String gender = (String) genComboBox.getSelectedItem();
            String days = typeTextField.getText();
            String accounttype = (String) reqLabelComboBox.getSelectedItem();

                //String qualification = qualificationField.getText();
                //String occupation = occupationField.getText();
                 String q1 = "insert into joinus values('" + name+ "', '" + phno + "','"+email+"','" + address + "', '" + gender + "','" +days + "','"+accounttype+"')";
            int x = stmt.executeUpdate(q1);
               if (x > 0)
               {
                System.out.println("Successfully Inserted");
               }
            else
                System.out.println("Insert Failed");
             }
            
                
               catch (ClassNotFoundException ex) {
                    Logger.getLogger(Project.class.getName()).log(Level.SEVERE, null, ex);
                } catch (SQLException ex) {
                    Logger.getLogger(Project.class.getName()).log(Level.SEVERE, null, ex);
                }
           
                // Perform database connectivity operations using the input values
                // Here, you can establish a connection to your Oracle database and perform the required operations
            }
            
        });
        }

      public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                new Project().setVisible(true);
            }
        });
      }
}