
import java.awt.Component;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.LinkedList;
import javax.swing.DefaultListModel;
import javax.swing.JOptionPane;

public class UserContacts extends javax.swing.JFrame {
    /**
     * Creates new form SetupContacts
     */
    public UserContacts() {
        initComponents();
        setExtendedState(start.MAXIMIZED_BOTH);
        contactList.setModel(start.dlm);
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        userContactsLabel = new javax.swing.JLabel();
        nameLabel = new javax.swing.JLabel();
        phoneLabel = new javax.swing.JLabel();
        cellLabel = new javax.swing.JLabel();
        cellProviderDrop = new javax.swing.JComboBox<String>();
        relationLabel = new javax.swing.JLabel();
        relationDrop = new javax.swing.JComboBox<String>();
        add = new javax.swing.JButton();
        delete = new javax.swing.JButton();
        jScrollPane2 = new javax.swing.JScrollPane();
        contactList = new javax.swing.JList<String>();
        MainMenu = new javax.swing.JButton();
        finish = new javax.swing.JButton();
        nameText = new javax.swing.JTextField();
        phoneNumberString = new javax.swing.JTextField();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        userContactsLabel.setFont(new java.awt.Font("Tahoma", 1, 36)); // NOI18N
        userContactsLabel.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        userContactsLabel.setText("User Contacts");

        nameLabel.setFont(new java.awt.Font("Tahoma", 0, 28)); // NOI18N
        nameLabel.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        nameLabel.setText("Name:");

        phoneLabel.setFont(new java.awt.Font("Tahoma", 0, 28)); // NOI18N
        phoneLabel.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        phoneLabel.setText("Phone \nNumber:");

        cellLabel.setFont(new java.awt.Font("Tahoma", 0, 28)); // NOI18N
        cellLabel.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        cellLabel.setText("Cell Provider:");

        cellProviderDrop.setFont(new java.awt.Font("Tahoma", 0, 28)); // NOI18N
        cellProviderDrop.setModel(new javax.swing.DefaultComboBoxModel(new String[] { "AT&T", "T-Mobile", "Sprint", "Verizon", "Nextel", "Virgin Mobile", "Cingular", "MetroPCS", "Cricket Wireless" }));

        relationLabel.setFont(new java.awt.Font("Tahoma", 0, 28)); // NOI18N
        relationLabel.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        relationLabel.setText("Relation:");

        relationDrop.setFont(new java.awt.Font("Tahoma", 0, 28)); // NOI18N
        relationDrop.setModel(new javax.swing.DefaultComboBoxModel(new String[] { "Spouse", "Sibling", "Friend", "Child", "Relative", "Parent" }));

        add.setFont(new java.awt.Font("Tahoma", 0, 28)); // NOI18N
        add.setText("Add");
        add.setActionCommand("Previous");
        add.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                addActionPerformed(evt);
            }
        });

        delete.setFont(new java.awt.Font("Tahoma", 0, 28)); // NOI18N
        delete.setText("Delete");
        delete.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                deleteActionPerformed(evt);
            }
        });

        jScrollPane2.setFont(new java.awt.Font("Tahoma", 0, 36)); // NOI18N

        contactList.setFont(new java.awt.Font("Tahoma", 0, 24)); // NOI18N
        contactList.setSelectionMode(javax.swing.ListSelectionModel.SINGLE_SELECTION);
        jScrollPane2.setViewportView(contactList);

        MainMenu.setFont(new java.awt.Font("Tahoma", 1, 30)); // NOI18N
        MainMenu.setText("Main Menu");
        MainMenu.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                MainMenuActionPerformed(evt);
            }
        });

        finish.setFont(new java.awt.Font("Tahoma", 1, 30)); // NOI18N
        finish.setText("Finish");
        finish.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                finishActionPerformed(evt);
            }
        });

        nameText.setFont(new java.awt.Font("Tahoma", 0, 24)); // NOI18N
        nameText.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                nameTextActionPerformed(evt);
            }
        });

        phoneNumberString.setFont(new java.awt.Font("Tahoma", 0, 24)); // NOI18N

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(layout.createSequentialGroup()
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                                        .addComponent(nameLabel)
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                        .addComponent(nameText, javax.swing.GroupLayout.PREFERRED_SIZE, 387, javax.swing.GroupLayout.PREFERRED_SIZE)
                                        .addGap(40, 40, 40))
                                    .addGroup(layout.createSequentialGroup()
                                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                            .addComponent(phoneLabel)
                                            .addComponent(cellLabel, javax.swing.GroupLayout.Alignment.TRAILING)
                                            .addComponent(relationLabel, javax.swing.GroupLayout.Alignment.TRAILING))
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                                            .addGroup(layout.createSequentialGroup()
                                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                                    .addComponent(cellProviderDrop, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                                    .addComponent(relationDrop, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                                                .addGap(104, 104, 104))
                                            .addGroup(layout.createSequentialGroup()
                                                .addComponent(phoneNumberString)
                                                .addGap(40, 40, 40)))))
                                .addComponent(jScrollPane2, javax.swing.GroupLayout.DEFAULT_SIZE, 218, Short.MAX_VALUE))
                            .addGroup(layout.createSequentialGroup()
                                .addGap(257, 257, 257)
                                .addComponent(userContactsLabel)
                                .addGap(0, 0, Short.MAX_VALUE)))
                        .addContainerGap())
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                        .addComponent(MainMenu, javax.swing.GroupLayout.PREFERRED_SIZE, 250, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(add)
                                .addGap(42, 42, 42)
                                .addComponent(delete))
                            .addComponent(finish, javax.swing.GroupLayout.PREFERRED_SIZE, 250, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addGap(23, 23, 23))))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap(31, Short.MAX_VALUE)
                .addComponent(userContactsLabel)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(nameLabel)
                            .addComponent(nameText, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addGap(16, 16, 16)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(phoneLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 56, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addGroup(layout.createSequentialGroup()
                                .addGap(7, 7, 7)
                                .addComponent(phoneNumberString, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)))
                        .addGap(18, 18, 18)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(cellProviderDrop, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(cellLabel))
                        .addGap(18, 18, 18)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(relationDrop, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(relationLabel)))
                    .addComponent(jScrollPane2, javax.swing.GroupLayout.PREFERRED_SIZE, 230, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(18, 18, 18)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(add)
                    .addComponent(delete))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, 32, Short.MAX_VALUE)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(MainMenu, javax.swing.GroupLayout.PREFERRED_SIZE, 65, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(finish, javax.swing.GroupLayout.PREFERRED_SIZE, 65, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addContainerGap())
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    
    private void addActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_addActionPerformed
        
        //making decisions about cell providers
        Object selectedItem = cellProviderDrop.getSelectedItem(); //look into value being reset on click??? possible error
        String cellCarrier = selectedItem.toString();
        String phoneNum = phoneNumberString.getText(); //look into value being reset on click??? possible error
        String smsGateway = "";

        cellCarrier = cellCarrier.toLowerCase();
        switch(cellCarrier) {
            case "at&t":
            smsGateway = "@txt.att.net";
            break;
            case "t-mobile":
            smsGateway = "@tmomail.net";
            break;
            case "sprint":
            smsGateway = "@messaging.sprintcs.com";
            break;
            case "verizon":
            smsGateway = "@vtext.com";
            break;
            case "nextel":
            smsGateway = "@messaging.nextel.com";
            break;
            case "virgin mobile":
            smsGateway = "@vmobl.com";
            break;
            case "cingular":
            smsGateway = "@cingularme.com";
            break;
            case "metropcs":
            smsGateway = "@mymetropcs.com";
            break;
            case "cricket wireless":
            smsGateway = "@sms.mycricket.com";
            break;
        }

        
            if((start.count>=5) && (start.deletedEntry<0)){
                  Component frame = null;
                    JOptionPane.showMessageDialog(frame,
                    "There is a 5 contact limit, please delete a contact before you add another");  
                    }
            else if(phoneNum.length() != 10 || !phoneNum.matches("[0-9]+")){
                  Component frame = null;
                    JOptionPane.showMessageDialog(frame,
                    "The phone number entered does not meet the proper format (10 numerical digits)!"); 
                    }
            else if(nameText.getText().isEmpty() || nameText.getText()==null){
                   Component frame = null;
                    JOptionPane.showMessageDialog(frame,
                    "Please enter your Name");    
                    
                }  
            else if(start.deletedEntry>=0 && start.deletedEntry<=4){
                String name = nameText.getText();
                Object item = relationDrop.getSelectedItem();
                String relation= item.toString();
                if(phoneNum.length() != 10 || !phoneNum.matches("[0-9]+")){
                  Component frame = null;
                    JOptionPane.showMessageDialog(frame,
                    "The phone number entered does not meet the proper format (10 numerical digits)!");  
                    }
                else if(nameText.getText().isEmpty() || nameText.getText()==null){
                   Component frame = null;
                    JOptionPane.showMessageDialog(frame,
                    "Please enter your Name");    
                    
                }  
                else{
                
                    Object cell = cellProviderDrop.getSelectedItem();
                    String cellPhone = cell.toString();
                    start.contacts[start.deletedEntry][0] = name;
                    start.contacts[start.deletedEntry][1] = relation;
                    start.contacts[start.deletedEntry][2] = phoneNum;
                    start.contacts[start.deletedEntry][3] = cellPhone;
                    start.contacts[start.deletedEntry][4] = phoneNum + smsGateway;
                    start.dlm.addElement(name);
                    start.deletedEntry = -1;
                }
            }
            else{
                               
                String name = nameText.getText();
                Object item = relationDrop.getSelectedItem();
                String relation= item.toString();
                Object cell = cellProviderDrop.getSelectedItem();
                String cellPhone = cell.toString();
                
                start.contacts[start.count][0] = name;
                start.contacts[start.count][1] = relation;
                start.contacts[start.count][2] = phoneNum;
                start.contacts[start.count][3] = cellPhone;
                start.contacts[start.count][4] = phoneNum + smsGateway;
                start.dlm.addElement(name);
                start.count = start.count + 1;
                System.out.println("This is the name: " + name);
            }
        
        contactList.setModel(start.dlm);
        
        
    }//GEN-LAST:event_addActionPerformed

    private void deleteActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_deleteActionPerformed
        start.deletedEntry = contactList.getSelectedIndex();
        start.dlm.removeElement(start.contacts[start.deletedEntry][0]);
        contactList.setModel(start.dlm);
        
        start.contacts[start.deletedEntry][4] = "wirelessinhomefalldetection@gmail.com"; 
        
        
    }//GEN-LAST:event_deleteActionPerformed

    private void MainMenuActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_MainMenuActionPerformed
          this.setVisible(false);
          new ControlPanelUser().setVisible(true);
        
    }//GEN-LAST:event_MainMenuActionPerformed

    private void finishActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_finishActionPerformed
        Component frame = null;
            JOptionPane.showMessageDialog(frame,
    "Great! Setup of contacts has been completed");
           this.setVisible(false);
        new ControlPanelUser().setVisible(true);
    }//GEN-LAST:event_finishActionPerformed

    private void nameTextActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_nameTextActionPerformed
    }//GEN-LAST:event_nameTextActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(UserContacts.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(UserContacts.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(UserContacts.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(UserContacts.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            @Override
            public void run() {
                new UserContacts().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton MainMenu;
    private javax.swing.JButton add;
    private javax.swing.JLabel cellLabel;
    private javax.swing.JComboBox<String> cellProviderDrop;
    private javax.swing.JList<String> contactList;
    private javax.swing.JButton delete;
    private javax.swing.JButton finish;
    private javax.swing.JScrollPane jScrollPane2;
    private javax.swing.JLabel nameLabel;
    private javax.swing.JTextField nameText;
    private javax.swing.JLabel phoneLabel;
    private javax.swing.JTextField phoneNumberString;
    private javax.swing.JComboBox<String> relationDrop;
    private javax.swing.JLabel relationLabel;
    private javax.swing.JLabel userContactsLabel;
    // End of variables declaration//GEN-END:variables
}
