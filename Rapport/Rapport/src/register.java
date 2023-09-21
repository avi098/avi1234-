import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.sql.*;
import java.util.logging.Level;
import java.util.logging.Logger;

public class register extends JFrame {
    private JPanel mainPanel;
    private JLabel titleLabel;
    private JLabel nameLabel;
    private JTextField nameField;
    private JLabel phLabel;
    private JTextField phField;
    private JLabel emailLabel;
    private JTextField emailField;
    private JLabel passwordLabel;
    private JPasswordField passwordField;
    private JLabel aadhaarLabel;
    private JTextField aadhaarField;
    private JLabel addressLabel;
    private JTextField addressField;
    private JLabel dobLabel;
    private JTextField dobField;
    private JLabel genderLabel;
    private JComboBox<String> genderComboBox;
    private JLabel reqLabel;
    private JComboBox<String> reqLabelComboBox;
    private JButton createAccountButton;


    public register (){
        setTitle("RAPPORT");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setResizable(false);
        initComponents();
        addComponents();
        addListeners();
        pack();
        setExtendedState(JFrame.MAXIMIZED_BOTH); // Set the JFrame to fullscreen
        setLocationRelativeTo(null); // Center the form on the screen
    }
    static void here()
    {
        System.out.println("Hello");
    }

    private void initComponents() {
        mainPanel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                // Load the background image
                Image backgroundImage = new ImageIcon("C:/Users/Shajithrooban/Downloads/hh3.jpg").getImage();
                // Draw the background image
                g.drawImage(backgroundImage, 0, 0, getWidth(), getHeight(), this);
            }
        };
        mainPanel.setLayout(new GridBagLayout());

        titleLabel = new JLabel("RAPPORT");
        titleLabel.setFont(new Font("Arial", Font.BOLD, 24));
        titleLabel.setForeground(Color.BLACK); // Set label text color to white

        nameLabel = new JLabel("Name:");
        nameLabel.setForeground(Color.BLACK); // Set label text color to white
        nameField = new JTextField();
        nameField.setOpaque(false); // Make text field background transparent

        emailLabel = new JLabel("Email:");
        emailLabel.setForeground(Color.BLACK); // Set label text color to white
        emailField = new JTextField();
        emailField.setColumns(20);
        emailField.setOpaque(false); // Make text field background transparent

        passwordLabel = new JLabel("Password:");
        passwordLabel.setForeground(Color.BLACK); // Set label text color to white
        passwordField = new JPasswordField();
        passwordField.setOpaque(false); // Make text field background transparent

        aadhaarLabel = new JLabel("Aadhaar No:");
        aadhaarLabel.setForeground(Color.BLACK); // Set label text color to white
        aadhaarField = new JTextField();
        aadhaarField.setOpaque(false); // Make text field background transparent

        addressLabel = new JLabel("Address:");
        addressLabel.setForeground(Color.BLACK); // Set label text color to white
        addressField = new JTextField();
        addressField.setOpaque(false); // Make text field background transparent

        dobLabel = new JLabel("UDID:");
        dobLabel.setForeground(Color.BLACK); // Set label text color to white
        dobField = new JTextField();
        dobField.setOpaque(false); // Make text field background transparent

        genderLabel = new JLabel("Gender:");
        genderLabel.setForeground(Color.BLACK); // Set label text color to white
        String[] genders = {"Male", "Female", "Other"};
        genderComboBox = new JComboBox<>(genders);
        genderComboBox.setOpaque(false); // Make combo box background transparent

        reqLabel = new JLabel("Requirement:");
        reqLabel.setForeground(Color.BLACK); // Set label text color to white
        String[] req = {"Wheel chair", "Hearing Aid", "Walking Cane","Proshetic Device","Employment","Medicene","Outfits"};
        reqLabelComboBox = new JComboBox<>(req);
        reqLabelComboBox.setOpaque(false); // Make combo box background transparent
        
        phLabel = new JLabel("Phone no:");
        phLabel.setForeground(Color.BLACK); // Set label text color to white
        phField = new JTextField();
        phField.setOpaque(false); // Make text field background transparent

        

        createAccountButton = new JButton("Register");
        createAccountButton.setBorderPainted(false);
        createAccountButton.setFocusPainted(false);
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.anchor = GridBagConstraints.WEST;
        gbc.insets = new Insets(10, 10, 20, 10);

        // Add components to the main panel
        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.gridwidth = 2;
        gbc.gridheight = 1;
        mainPanel.add(titleLabel, gbc);

        gbc.gridx = 0;
        gbc.gridy = 1;
        gbc.gridwidth = 1;
        mainPanel.add(nameLabel, gbc);

        gbc.gridx = 1;
        gbc.gridy = 1;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        mainPanel.add(nameField, gbc);

        gbc.gridx = 0;
        gbc.gridy = 2;
        gbc.fill = GridBagConstraints.NONE;
        mainPanel.add(emailLabel, gbc);

        gbc.gridx = 1;
        gbc.gridy = 2;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        mainPanel.add(emailField, gbc);

        gbc.gridx = 0;
        gbc.gridy = 3;
        gbc.fill = GridBagConstraints.NONE;
        mainPanel.add(passwordLabel, gbc);

        gbc.gridx = 1;
        gbc.gridy = 3;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        mainPanel.add(passwordField, gbc);

        gbc.gridx = 0;
        gbc.gridy = 4;
        gbc.fill = GridBagConstraints.NONE;
        mainPanel.add(aadhaarLabel, gbc);

        gbc.gridx = 1;
        gbc.gridy = 4;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        mainPanel.add(aadhaarField, gbc);

        gbc.gridx = 0;
        gbc.gridy = 5;
        gbc.fill = GridBagConstraints.NONE;
        mainPanel.add(addressLabel, gbc);

        gbc.gridx = 1;
        gbc.gridy = 5;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        mainPanel.add(addressField, gbc);

        gbc.gridx = 0;
        gbc.gridy = 6;
        gbc.gridwidth = 2;
        gbc.gridheight = 1;
        gbc.fill = GridBagConstraints.NONE;
        mainPanel.add(dobLabel, gbc);

        gbc.gridx = 1;
        gbc.gridy = 6;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        mainPanel.add(dobField, gbc);

        gbc.gridx = 0;
        gbc.gridy = 7;
        gbc.fill = GridBagConstraints.NONE;
        mainPanel.add(genderLabel, gbc);

        gbc.gridx = 1;
        gbc.gridy = 7;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        mainPanel.add(genderComboBox, gbc);

        gbc.gridx = 0;
        gbc.gridy = 8;
        gbc.fill = GridBagConstraints.NONE;
        mainPanel.add(reqLabel, gbc);

        gbc.gridx = 1;
        gbc.gridy = 8;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        mainPanel.add(reqLabelComboBox, gbc);
        
        gbc.gridx = 0;
        gbc.gridy = 10;
        gbc.gridwidth = 2;
        gbc.gridheight = 1;
        gbc.fill = GridBagConstraints.NONE;
        mainPanel.add(phLabel, gbc);

        gbc.gridx = 1;
        gbc.gridy = 10;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        mainPanel.add(phField, gbc);

        gbc.gridx = 0;
        gbc.gridy = 11;
        gbc.gridwidth = 2;
        gbc.fill = GridBagConstraints.NONE;
        mainPanel.add(createAccountButton, gbc);
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
        createAccountButton.addActionListener(new ActionListener() {
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
            
   
                String name = nameField.getText();
                String email = emailField.getText();
                String password = new String(passwordField.getPassword());
                String aadhaarNo = aadhaarField.getText();
                String address = addressField.getText();
                String dob = dobField.getText();
                String gender = (String) genderComboBox.getSelectedItem();
                String accountType = (String) reqLabelComboBox.getSelectedItem();
                String pno = phField.getText();
                //String qualification = qualificationField.getText();
                //String occupation = occupationField.getText();
                 String q1 = "insert into register values('" + name+ "', '" + email + "', '" + password + "', '" + aadhaarNo + "','" +address + "','"+dob+"','"+gender+"','"+accountType+"','"+pno+"')";
            int x = stmt.executeUpdate(q1);
               if (x > 0)
                System.out.println("Successfully Inserted");
            else
                System.out.println("Insert Failed");
             }
            
                
               catch (ClassNotFoundException ex) {
                    Logger.getLogger(register.class.getName()).log(Level.SEVERE, null, ex);
                } catch (SQLException ex) {
                    Logger.getLogger(register.class.getName()).log(Level.SEVERE, null, ex);
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
                new register().setVisible(true);
            }
        });
    }
}